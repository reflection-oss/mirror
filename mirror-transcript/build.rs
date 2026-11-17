#![forbid(unsafe_code)]

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=transcript.proto");
    prost_build::Config::new()
        .boxed(".mirror.tuir.v1.UiCommand.payload.selector")
        .boxed(".mirror.tuir.v1.ChromeFrame.update")
        .compile_protos(&["transcript.proto"], &["."])
}
