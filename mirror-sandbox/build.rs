use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let staging = PathBuf::from(std::env::var("OUT_DIR")?).join("protos");
    std::fs::create_dir_all(&staging)?;
    for source in [
        "sandboxd.proto",
        "../protocols/mirror_agent/protos/agent.proto",
        "../protocols/mirror_agent/protos/mirror.proto",
    ] {
        println!("cargo:rerun-if-changed={source}");
        // Cargo and Bazel use different import roots for these shared schemas.
        let contents = std::fs::read_to_string(source)?
            .replace("protocols/mirror_agent/protos/", "")
            .replace("mirror/protocol/", "");
        let name = PathBuf::from(source);
        std::fs::write(
            staging.join(name.file_name().ok_or("proto source needs a filename")?),
            contents,
        )?;
    }
    let mut includes = vec![staging.clone()];
    if let Ok(well_known_types) = std::env::var("PROTOBUF_WELL_KNOWN_TYPES") {
        let well_known = well_known_types
            .split_whitespace()
            .next()
            .ok_or("PROTOBUF_WELL_KNOWN_TYPES must contain a protobuf source path")?;
        let protobuf_root = PathBuf::from(well_known)
            .ancestors()
            .nth(3)
            .ok_or("well-known protobuf source path must have three parent directories")?
            .to_path_buf();
        includes.push(protobuf_root);
    }
    tonic_prost_build::configure()
        // Exported agent protos use the Python package's import prefix.
        .protoc_arg("--proto_path=mirror/protocol=../protocols/mirror_agent/protos")
        .skip_debug([
            ".mirror.sandbox.v1.EgressIdentity",
            ".mirror.sandbox.v1.EgressUpdate",
        ])
        .extern_path(".mirror.agent.v1", "::mirror_protocol")
        .extern_path(".mirror.v1", "::mirror_protocol::mirror")
        .boxed(".mirror.sandbox.v1.ExecuteResponse.outcome.completed")
        .compile_protos(&[staging.join("sandboxd.proto")], &includes)?;
    Ok(())
}
