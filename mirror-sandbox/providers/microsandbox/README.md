# mirror-sandbox-microsandbox

Runs Mirror sandboxes as [microsandbox](https://microsandbox.dev) microVMs. It
implements `SandboxProvider` and `SandboxTransport` from `mirror-sandbox` by
calling the `msb` command.

Mirror does not ship microsandbox. Install `msb` 0.7.7 or newer yourself; the
provider uses `$MIRROR_MSB` if set and the first `msb` on `PATH` otherwise.
macOS needs Apple silicon. Linux needs read and write access to `/dev/kvm`.

Mirror's wheels do ship `sandboxd`: `mirror/bin/linux/mirror-sandboxd` is built
for the wheel's architecture and linked against glibc 2.17, so it runs in any
glibc-based image; Alpine and other musl images are not supported. `build.sh`
cross-links it with Zig, which needs the matching Rust target
(`rustup target add aarch64-unknown-linux-gnu` on Apple silicon).

## What a sandbox looks like

```text
mirror-ws-<request>                 one VM per provisioning request
├── labels      mirror.workspace, mirror.request=<id>, mirror.session=<id>
├── network     deny by default; HTTPS (443) to the configured hosts only
├── <workspace> the host directory, read-write, at the same path
└── /usr/local/bin/mirror-sandboxd   the host's Linux build, read-only
```

- Every use first inspects the VM. A VM that is not running, whose network
  policy is not exactly the configured allowlist, or whose mounts are not this
  workspace and one `sandboxd` file, is refused rather than used. `msb exec` on
  a stopped VM would otherwise boot a throwaway copy. `create` removes a VM
  with the wrong network policy, and one it just made with the wrong mounts.
- Workspace tools reach `sandboxd serve` over `msb exec --stream`.
- The guest kernel can keep stale metadata for files the host changes, so the
  provider asks `sandboxd` to drop the guest's dentry and inode caches before
  each call. This needs the image to run commands as root.
- Suspending stops the VM. Its root filesystem survives until the sandbox is
  deleted, and the workspace lives on the host.

## Tests

The unit tests drive a fake `msb` (`testdata/fake_msb.sh`). The live test boots
real VMs and is manual:

```bash
MIRROR_LIVE_SANDBOXD=/path/to/linux/mirror-sandboxd \
  cargo test -p mirror-sandbox-microsandbox --test live_acceptance -- --ignored
```

`MIRROR_LIVE_SANDBOXD` must be built for the host's architecture.
`MIRROR_LIVE_IMAGE` overrides the default `debian:bookworm-slim`.
