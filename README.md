# Mirror

A coding-agent runtime and terminal client.

- [`mirror-agent`](mirror-agent/README.md): Rust runtime and HTTP server.
- [`mirror-cli`](mirror-cli/README.md): Python terminal client and extensions.
- [`protocols/mirror_agent`](protocols/mirror_agent): shared protobuf schemas.
- [`mirror-agent-py`](mirror-agent-py/README.md): Python bindings for embedding.

## Build and run

Install Rust/Cargo, protoc, Python 3.12+, uv and ripgrep, then run:

```bash
cargo build --locked -p mirror-agent --bin mirror-agent
MIRROR_RUST_WORKER="$PWD/target/debug/mirror-agent" \
  uv run --locked --project mirror-cli mirror --help
```

See the [CLI guide](mirror-cli/README.md) for provider and model configuration.
The PyPI project named `mirror` is unrelated; install this project from source.
Protocol bindings are generated during the build. Keep the workspace components
and protocol directory together.

Tools run with your account's permissions. Mirror is not a sandbox; use an
isolated environment for untrusted projects or agent-controlled execution.

## Test

```bash
bash ci.sh
cargo test --locked -p mirror-agent -p mirror-protocol
cargo check --locked -p mirror-agent-python
```

## License

[Apache-2.0](LICENSE). Preserve the component notices and third-party licenses
when redistributing source or binaries.
