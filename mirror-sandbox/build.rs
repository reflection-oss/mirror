use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=sandboxd.proto");
    // `sandboxd.proto` imports `agent.proto` relative to the enclosing
    // repository root; the standalone export rewrites that import relative to
    // this workspace, whose root is the parent directory.
    let mut includes: Vec<PathBuf> = [".", "..", "../../../.."]
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .collect();
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
        .skip_debug([
            ".mirror.sandbox.v1.EgressIdentity",
            ".mirror.sandbox.v1.EgressUpdate",
        ])
        .extern_path(".mirror.agent.v1", "::mirror_protocol")
        .boxed(".mirror.sandbox.v1.ExecuteResponse.outcome.completed")
        .compile_protos(&[PathBuf::from("sandboxd.proto")], &includes)?;
    Ok(())
}
