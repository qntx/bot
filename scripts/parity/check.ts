/// <reference types="node" />
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Validates parity.json: capability ids, referenced files, difference
// classes, and that every vector case's classes are declared by its
// capability. `checkParity` is pure and takes an injected filesystem so the
// failure modes are testable; the CLI below wires it to node:fs.

type JsonObject = Record<string, unknown>;

export type ParityFs = {
  fileExists: (path: string) => boolean;
  fileText: (path: string) => string | undefined;
};

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function vectorCaseClasses(fs: ParityFs, path: string): string[][] {
  const text = fs.fileText(path);
  if (text === undefined) {
    return [];
  }
  let document: unknown;
  try {
    document = JSON.parse(text);
  } catch {
    return [];
  }
  const cases = isRecord(document) ? document["cases"] : undefined;
  if (!Array.isArray(cases)) {
    return [];
  }
  return cases.map((entry) => (isRecord(entry) ? asStringArray(entry["classes"]) : []));
}

const ID_PATTERN = /^[A-Z]+(-[A-Z0-9]+)+$/;

/** Validate a parsed parity.json; returns one error string per violation. */
export function checkParity(parity: unknown, fs: ParityFs, upstream = false): string[] {
  const errors: string[] = [];
  const root = isRecord(parity) ? parity : {};
  const classes = isRecord(root["classes"]) ? root["classes"] : {};
  const definedClasses = new Set(Object.keys(classes));
  const usedClasses = new Set<string>();
  const capabilities = Array.isArray(root["capabilities"]) ? root["capabilities"] : [];
  const seenIds = new Set<string>();

  for (const capability of capabilities) {
    const entry = isRecord(capability) ? capability : {};
    const id = typeof entry["id"] === "string" ? entry["id"] : "<missing>";
    if (typeof entry["id"] === "string") {
      if (!ID_PATTERN.test(id)) {
        errors.push(`${id}: id must match ${ID_PATTERN.source}`);
      }
      if (seenIds.has(id)) {
        errors.push(`${id}: duplicate capability id`);
      }
      seenIds.add(id);
    } else {
      errors.push("capability: missing string id");
    }

    const capabilityClasses = asStringArray(entry["classes"]);
    for (const name of capabilityClasses) {
      usedClasses.add(name);
      if (!definedClasses.has(name)) {
        errors.push(`${id}: class "${name}" is not defined in parity.json classes`);
      }
    }
    const classSet = new Set(capabilityClasses);

    for (const path of asStringArray(entry["vectors"])) {
      if (!fs.fileExists(path)) {
        errors.push(`${id}: vector file missing: ${path}`);
        continue;
      }
      for (const caseClasses of vectorCaseClasses(fs, path)) {
        for (const name of caseClasses) {
          if (!classSet.has(name)) {
            errors.push(`${id}: vector ${path} case uses undeclared class "${name}"`);
          }
        }
      }
    }

    for (const path of asStringArray(entry["rust"])) {
      if (!fs.fileExists(path)) {
        errors.push(`${id}: rust file missing: ${path}`);
        continue;
      }
      const text = fs.fileText(path) ?? "";
      const escaped = id.replaceAll(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
      const marker = new RegExp(`parity: ${escaped}(?![A-Z0-9-])`);
      if (!marker.test(text)) {
        errors.push(`${id}: rust file lacks "parity: ${id}" marker: ${path}`);
      }
    }

    if (upstream) {
      for (const path of asStringArray(entry["upstream"])) {
        if (!fs.fileExists(join("3rdparty", "pi", path))) {
          errors.push(`${id}: upstream file missing: ${path}`);
        }
      }
    }
  }

  for (const name of definedClasses) {
    if (!usedClasses.has(name)) {
      errors.push(`class "${name}" is defined but never used`);
    }
  }

  return errors;
}

const MILESTONE_PATTERN = /^M[0-9]$/;

/** Why an owner entry is invalid, or `undefined` when it is valid. */
function ownerError(owner: unknown, capabilities: Set<string>): string | undefined {
  if (Array.isArray(owner)) {
    if (owner.length === 0) {
      return "lists no capability";
    }
    const unknown = owner.filter((id) => typeof id !== "string" || !capabilities.has(id));
    return unknown.length > 0
      ? `names capabilities not in parity.json: ${unknown.join(", ")}`
      : undefined;
  }
  if (!isRecord(owner)) {
    return "must be a capability list or an object";
  }
  if (typeof owner["e2e"] === "string" || typeof owner["excluded"] === "string") {
    return Object.keys(owner).length === 1 ? undefined : "takes no other fields";
  }
  const { milestone } = owner;
  if (typeof milestone !== "string" || !MILESTONE_PATTERN.test(milestone)) {
    return 'needs "e2e", "excluded" or a "milestone" like "M5"';
  }
  const { capability } = owner;
  if (
    capability !== undefined &&
    (typeof capability !== "string" || !ID_PATTERN.test(capability))
  ) {
    return `capability must match ${ID_PATTERN.source}`;
  }
  return undefined;
}

/**
 * Validate `parity.json` `upstreamTests` against the upstream test inventory: the inventory is of
 * the pinned commit, every listed file has an owner (its own entry, else the longest directory
 * entry ending in `/` above it), every entry owns a listed file, and every owner is valid.
 */
export function checkUpstreamTests(parity: unknown, inventory: unknown): string[] {
  const errors: string[] = [];
  const root = isRecord(parity) ? parity : {};
  const commit = isRecord(root["upstream"]) ? root["upstream"]["commit"] : undefined;
  const listed = isRecord(inventory) ? inventory : {};
  if (listed["commit"] !== commit) {
    errors.push(
      `upstream test inventory is of ${String(listed["commit"])}, parity.json pins ${String(commit)}`,
    );
  }
  const files = Object.keys(isRecord(listed["files"]) ? listed["files"] : {});
  const owners = isRecord(root["upstreamTests"]) ? root["upstreamTests"] : {};
  const capabilities = new Set(
    (Array.isArray(root["capabilities"]) ? root["capabilities"] : [])
      .map((entry) => (isRecord(entry) ? entry["id"] : undefined))
      .filter((id): id is string => typeof id === "string"),
  );
  const used = new Set<string>();
  for (const file of files) {
    const key =
      file in owners
        ? file
        : Object.keys(owners)
            .filter((entry) => entry.endsWith("/") && file.startsWith(entry))
            .toSorted((a, b) => b.length - a.length)[0];
    if (key === undefined) {
      errors.push(`upstream test ${file} has no owner in parity.json upstreamTests`);
    } else {
      used.add(key);
    }
  }
  for (const [key, owner] of Object.entries(owners)) {
    if (!used.has(key)) {
      errors.push(`upstreamTests entry ${key} owns no listed upstream test`);
    }
    const error = ownerError(owner, capabilities);
    if (error !== undefined) {
      errors.push(`upstreamTests entry ${key} ${error}`);
    }
  }
  return errors;
}

if (import.meta.main) {
  const upstream = process.argv.includes("--upstream");
  const parity = JSON.parse(readFileSync("parity.json", "utf8")) as unknown;
  const fs: ParityFs = {
    fileExists: (path) => existsSync(path),
    fileText: (path) => (existsSync(path) ? readFileSync(path, "utf8") : undefined),
  };
  const inventory = JSON.parse(readFileSync("vectors/upstream/tests.json", "utf8")) as unknown;
  const errors = [...checkParity(parity, fs, upstream), ...checkUpstreamTests(parity, inventory)];
  for (const error of errors) {
    console.error(`parity: ${error}`);
  }
  if (errors.length > 0) {
    process.exitCode = 1;
  }
}
