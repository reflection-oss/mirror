//! Native setup for Mirror's consent-installed sandbox assets and private runtime state.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use mirror_protocol::WorkerSandbox;
use tonic::Status;

use crate::AgentOptions;

/// This layout is shared by installers, not supplied by guest-writable workspace metadata.
fn installation(options: &AgentOptions) -> PathBuf {
    let platform = if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    };
    options.home.join(".mirror/sandbox-assets").join(format!(
        "microsandbox-0.7.7-{platform}-{}",
        std::env::consts::ARCH
    ))
}

fn library_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libkrunfw.dylib"
    } else {
        "libkrunfw.so"
    }
}

pub fn sandbox_command(options: &AgentOptions) -> tokio::process::Command {
    let root = options.home.join(".mirror/msb");
    let installation = installation(options);
    let mut command = super::workspace_command(options, &options.executable, &options.home);
    command
        .env("MIRROR_MSB", installation.join("msb"))
        .env("MSB_HOME", &root)
        .env("MSB_CONFIG_PATH", root.join("config.json"))
        .env("MSB_LIBKRUNFW_PATH", installation.join(library_name()));
    command
}

/// Encode the native sandbox helper's configuration from the worker's canonical inputs.
pub fn sandbox_configuration_json(sandbox: &WorkerSandbox) -> Result<String, Status> {
    let source = sandbox
        .source
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("sandbox source is required"))?;
    Ok(serde_json::json!({
        "id": sandbox.id,
        "image": source.image,
        "sandboxd": sandbox.sandboxd,
        "egress": source.egress,
        "cpus": source.cpus,
        "memory_mib": source.memory_mib,
    })
    .to_string())
}

/// Create only private host configuration; downloading and VM startup require separate consent.
pub async fn prepare_runtime(options: &AgentOptions) -> Result<(), Status> {
    let options = options.clone();
    tokio::task::spawn_blocking(move || setup(&options)).await
        .map_err(|_| Status::internal("sandbox runtime setup task failed"))?
        .map_err(|_| Status::failed_precondition("sandbox runtime is unavailable; install the approved native assets and check private state directories"))
}

fn setup(options: &AgentOptions) -> io::Result<()> {
    let managed = if cfg!(target_os = "macos") {
        "/Library/Application Support/microsandbox/managed.json"
    } else {
        "/etc/microsandbox/managed.json"
    };
    if Path::new(managed).try_exists()? {
        return Err(io::Error::other(
            "machine-wide Microsandbox configuration is unsupported",
        ));
    }
    let installation = installation(options);
    if !installation.join("msb").is_file() || !installation.join(library_name()).is_file() {
        return Err(io::Error::other("sandbox runtime assets are missing"));
    }
    let root = options.home.join(".mirror/msb");
    let cache = options.home.join(".mirror/sandbox-assets/cache");
    crate::private_directory(&root)?;
    crate::private_directory(&cache)?;
    let mut config = tempfile::NamedTempFile::new_in(&root)?;
    config.write_all(
        serde_json::json!({"paths": {"cache": cache}})
            .to_string()
            .as_bytes(),
    )?;
    config.as_file().sync_all()?;
    config
        .persist(root.join("config.json"))
        .map_err(|error| error.error)?;
    Ok(())
}
