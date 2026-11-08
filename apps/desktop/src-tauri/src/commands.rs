//! Typed commands exposed to the frontend over tauri-specta IPC.

use serde::Serialize;

/// Identity reported by the backend at startup.
#[derive(Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AppInfo {
    /// Product name from the app package metadata.
    pub name: String,
    /// Version from the app package metadata.
    pub version: String,
}

/// Returns the app name and version from the app package metadata.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Tauri injects AppHandle by value into command signatures"
)]
#[tauri::command]
#[specta::specta]
pub(crate) fn app_info(app: tauri::AppHandle) -> AppInfo {
    let info = app.package_info();
    AppInfo {
        name: info.name.clone(),
        version: info.version.to_string(),
    }
}
