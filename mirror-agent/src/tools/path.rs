//! Workspace path resolution and tool-schema helpers.

use std::fs;
use std::io;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use mirror_protocol::TouchedPath;

use super::ToolError;

pub(crate) fn canonical_workspace(root: impl AsRef<Path>) -> Result<PathBuf, ToolError> {
    fs::canonicalize(root).map_err(execution_error)
}

pub(crate) async fn resolve_existing(
    root: &Path,
    path: impl AsRef<Path>,
    allow_outside_workspace: bool,
) -> Result<PathBuf, ToolError> {
    let path = path.as_ref();
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let resolved = tokio::fs::canonicalize(candidate)
        .await
        .map_err(execution_error)?;
    ensure_contained(root, &resolved, allow_outside_workspace)?;
    Ok(resolved)
}

pub(crate) async fn resolve_for_write(
    root: &Path,
    path: impl AsRef<Path>,
    allow_outside_workspace: bool,
) -> Result<PathBuf, ToolError> {
    let path = path.as_ref();
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let mut existing = candidate.as_path();
    loop {
        match tokio::fs::symlink_metadata(existing).await {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                existing = existing.parent().ok_or_else(|| {
                    ToolError::Execution(format!(
                        "path {} has no existing ancestor",
                        candidate.display()
                    ))
                })?;
            }
            Err(error) => return Err(execution_error(error)),
        }
    }
    let resolved_parent = tokio::fs::canonicalize(existing)
        .await
        .map_err(execution_error)?;
    let suffix = candidate
        .strip_prefix(existing)
        .map_err(|error| ToolError::Execution(error.to_string()))?;
    let resolved = normalize_lexically(&resolved_parent.join(suffix));
    ensure_contained(root, &resolved, allow_outside_workspace)?;
    Ok(resolved)
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component),
        }
    }
    normalized
}

fn ensure_contained(
    root: &Path,
    path: &Path,
    allow_outside_workspace: bool,
) -> Result<(), ToolError> {
    if !allow_outside_workspace && !path.starts_with(root) {
        return Err(ToolError::Execution(format!(
            "path {} is outside workspace {}",
            path.display(),
            root.display()
        )));
    }
    Ok(())
}

pub(crate) fn relative_display(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The fact that a tool touched `path`, reported relative to `root`.
pub(crate) fn touched(root: &Path, path: &Path, kind: TouchedKind) -> TouchedPath {
    let mut relative = relative_display(path, root);
    if relative.is_empty() {
        relative.push('.');
    }
    TouchedPath {
        path: relative,
        kind: kind as i32,
    }
}

pub(crate) fn execution_error(error: std::io::Error) -> ToolError {
    ToolError::Execution(error.to_string())
}

pub(crate) fn object_schema(
    properties: serde_json::Map<String, serde_json::Value>,
    required: &[&str],
) -> serde_json::Map<String, serde_json::Value> {
    serde_json::Map::from_iter([
        ("type".to_string(), serde_json::json!("object")),
        ("properties".to_string(), properties.into()),
        ("required".to_string(), serde_json::json!(required)),
        ("additionalProperties".to_string(), serde_json::json!(false)),
    ])
}
