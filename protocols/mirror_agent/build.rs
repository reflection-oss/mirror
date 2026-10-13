use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = ["protos/agent.proto", "protos/mirror.proto"];
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    let mut includes = vec![PathBuf::from("protos")];
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
    let mut config = tonic_prost_build::Config::new();
    config.enable_type_names();
    tonic_prost_build::configure()
        .file_descriptor_set_path(PathBuf::from(std::env::var("OUT_DIR")?).join("descriptor.bin"))
        .boxed(".mirror.agent.v1.ClientMessage.payload.message")
        .boxed(".mirror.agent.v1.ChatRecord.sampling")
        .type_attribute(
            ".mirror.agent.v1.SessionMessage.payload",
            "#[allow(clippy::large_enum_variant)]",
        )
        .compile_with_config(config, &protos.map(PathBuf::from), &includes)?;
    Ok(())
}
