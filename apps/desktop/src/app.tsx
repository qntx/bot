import { useEffect, useState } from "react";
import type { JSX } from "react";

import { commands } from "./bindings.gen.ts";
import type { AppInfo } from "./bindings.gen.ts";

export function App(): JSX.Element {
  const [info, setInfo] = useState<AppInfo>();

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setInfo(await commands.appInfo());
      } catch {
        // Not running inside the Tauri webview (plain `vp dev` in a browser).
      }
    };
    void load();
  }, []);

  return <main>{info === undefined ? "…" : `${info.name} ${info.version}`}</main>;
}
