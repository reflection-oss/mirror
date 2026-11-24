use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let staging = PathBuf::from(std::env::var("OUT_DIR")?).join("protos");
    std::fs::create_dir_all(&staging)?;
    let protos = [
        "agent.proto",
        "mirror.proto",
        "daemon.proto",
        "local_models.proto",
    ];
    for proto in protos {
        let source = PathBuf::from("protos").join(proto);
        println!("cargo:rerun-if-changed={}", source.display());
        // Cargo and Bazel share descriptors despite their different import roots.
        let contents = std::fs::read_to_string(source)?
            .replace("protocols/mirror_agent/protos/", "")
            .replace("mirror/protocol/", "");
        std::fs::write(staging.join(proto), contents)?;
    }
    let mut includes = vec![staging.clone()];
    if let Ok(well_known_types) = std::env::var("PROTOBUF_WELL_KNOWN_TYPES") {
        let well_known_type = well_known_types.split_whitespace().next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "PROTOBUF_WELL_KNOWN_TYPES must contain at least one source path",
            )
        })?;
        let protobuf_src = PathBuf::from(well_known_type)
            .ancestors()
            .nth(3)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "well-known protobuf source path {well_known_type:?} must have at least three parent directories"
                    ),
                )
            })?
            .to_path_buf();
        includes.push(protobuf_src);
    }
    // Generate importers first: tonic also emits services for their imported packages.
    let mut daemon_config = tonic_prost_build::Config::new();
    daemon_config.enable_type_names();
    tonic_prost_build::configure()
        .extern_path(".mirror.agent.v1", "crate")
        .extern_path(".mirror.v1", "crate::mirror")
        .file_descriptor_set_path(PathBuf::from(std::env::var("OUT_DIR")?).join("descriptor.bin"))
        .compile_with_config(daemon_config, &[staging.join("daemon.proto")], &includes)?;
    let mut config = tonic_prost_build::Config::new();
    config.enable_type_names();
    tonic_prost_build::configure()
        .extern_path(".mirror.v1", "crate::mirror")
        .boxed(".mirror.agent.v1.ClientMessage.payload.message")
        .boxed(".mirror.agent.v1.ChatRecord.sampling")
        .type_attribute(
            ".mirror.agent.v1.SessionMessage.payload",
            "#[allow(clippy::large_enum_variant)]",
        )
        .type_attribute(
            ".mirror.agent.v1.AgentMessage.payload",
            "#[allow(clippy::large_enum_variant)]",
        )
        .compile_with_config(config, &[staging.join("agent.proto")], &includes)?;
    let mut mirror_config = tonic_prost_build::Config::new();
    mirror_config.enable_type_names();
    tonic_prost_build::configure()
        .boxed(".mirror.v1.WatchSessionsResponse.update.event")
        .compile_with_config(
            mirror_config,
            &[
                staging.join("mirror.proto"),
                staging.join("local_models.proto"),
            ],
            &includes,
        )?;
    Ok(())
}
