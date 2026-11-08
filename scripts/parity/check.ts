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
      if (!text.includes(`parity: ${id}`)) {
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

if (import.meta.main) {
  const upstream = process.argv.includes("--upstream");
  const parity = JSON.parse(readFileSync("parity.json", "utf8")) as unknown;
  const fs: ParityFs = {
    fileExists: (path) => existsSync(path),
    fileText: (path) => (existsSync(path) ? readFileSync(path, "utf8") : undefined),
  };
  const errors = checkParity(parity, fs, upstream);
  for (const error of errors) {
    console.error(`parity: ${error}`);
  }
  if (errors.length > 0) {
    process.exitCode = 1;
  }
}
