# Develop from source

Use the full Mirror checkout, not a Python source archive. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), Rust, a C compiler,
`protoc`, and ripgrep (`rg`). Run these commands from the checkout root.

## Run

```bash
./mirror-cli/mirror-dev
```

**Check:** the TUI opens. The script builds the Rust worker and runs the Python
CLI from your checkout. It does not replace your installed `mirror`.

## Make a change

Edit `mirror-cli/src/mirror/` for the CLI or `mirror-agent/src/` for the worker.
Exit and rerun `mirror-dev` to rebuild and try the change.

For UI-only work, skip the worker build and use a synthetic conversation:

```bash
./mirror-cli/mirror-dev --showcase
```

**Check:** sending a message plays the demo without model calls or real tools.

## Run checks

```bash
bash mirror-cli/ci.sh tests/cli/main_test.py
```

**Check:** type checking and the selected tests pass. Replace the test path with
your changed tests; omit it to run the full Python suite.

## Build a wheel

Only needed to distribute or install your changes:

```bash
bash mirror-cli/build.sh ./dist
```

**Check:** a platform wheel appears in `dist/` after its install checks pass.
