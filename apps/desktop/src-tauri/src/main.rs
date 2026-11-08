//! Desktop shell for the ovo agent harness.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use commands::{__specta__fn__app_info, app_info};

/// The specta builder is shared between `invoke_handler` and the bindings
/// test, so the exported types always match what the app registers.
#[expect(
    trivial_casts,
    reason = "collect_commands! emits a redundant fn-pointer cast"
)]
fn specta_builder() -> tauri_specta::Builder<tauri::Wry> {
    tauri_specta::Builder::<tauri::Wry>::new().commands(tauri_specta::collect_commands![app_info])
}

#[expect(
    clippy::disallowed_methods,
    reason = "tauri::generate_context! calls std::process::exit on a malformed config"
)]
fn main() -> tauri::Result<()> {
    tauri::Builder::default()
        .invoke_handler(specta_builder().invoke_handler())
        .run(tauri::generate_context!())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::process;

    use super::specta_builder;

    const BINDINGS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/bindings.gen.ts");

    fn render_bindings() -> String {
        // tauri_specta only exports to a path; render into a temp file so the
        // committed file can be compared byte-for-byte.
        let tmp = env::temp_dir().join(format!("ovo-desktop-bindings-{}.ts", process::id()));
        specta_builder()
            .export(specta_typescript::Typescript::default(), &tmp)
            .expect("typescript bindings export failed");
        let rendered = fs::read_to_string(&tmp).expect("failed to read exported bindings");
        fs::remove_file(&tmp).expect("failed to remove temp bindings file");
        rendered
    }

    /// `bindings.gen.ts` is generated here and nowhere else: set
    /// `OVO_UPDATE_BINDINGS=1` to rewrite it after changing a command.
    #[test]
    fn bindings_are_current() {
        let generated = render_bindings();
        if env::var("OVO_UPDATE_BINDINGS").is_ok_and(|value| value == "1") {
            fs::write(BINDINGS_PATH, &generated).expect("failed to write bindings.gen.ts");
            return;
        }
        let committed = fs::read_to_string(BINDINGS_PATH).expect(
            "src/bindings.gen.ts is missing; regenerate with OVO_UPDATE_BINDINGS=1 cargo test",
        );
        assert_eq!(
            generated, committed,
            "src/bindings.gen.ts is stale; regenerate with OVO_UPDATE_BINDINGS=1 cargo test"
        );
    }
}
