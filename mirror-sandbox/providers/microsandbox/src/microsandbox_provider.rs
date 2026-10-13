//! Sandbox lifecycle on microsandbox microVMs.

use std::ffi::OsString;
use std::time::Duration;

use async_trait::async_trait;
use mirror_sandbox::SandboxLiveness;
use mirror_sandbox::SandboxProvider;
use mirror_sandbox::SandboxProviderError;
use tokio::time::Instant;
use uuid::Uuid;

use crate::InvalidMicrosandboxConfig;
use crate::MicrosandboxConfig;
use crate::Msb;
use crate::MsbFailure;
use crate::SandboxInspection;
use crate::SandboxStatus;
use crate::REQUEST_LABEL;

/// Runs each sandbox as a microVM named `mirror-ws-<request>`. Suspending
/// stops the VM; its root filesystem and the host workspace survive until the
/// sandbox is deleted.
#[derive(Debug)]
pub struct MicrosandboxProvider {
    // Crate-visible for the `SandboxTransport` implementation beside this one.
    pub(crate) msb: Msb,
    pub(crate) config: MicrosandboxConfig,
}

/// What a poll step decides: keep waiting (`None`) or finish with a result.
type PollStep<T> = Option<Result<T, SandboxProviderError>>;

/// Whether a sandbox existed before the `create` call that adopts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Provenance {
    /// It existed already or another create made it first. It may serve
    /// another workspace, so its mounts alone never get it removed.
    Existing,
    /// This call made it, even if `msb create` failed or timed out after
    /// making it, so a retry can only succeed once it is removed.
    New,
}

/// `msb create` may pull the image first.
const CREATE_TIMEOUT: Duration = Duration::from_secs(600);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const READY_TIMEOUT: Duration = Duration::from_secs(120);
const SUSPEND_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(200);

impl MicrosandboxProvider {
    pub fn new(msb: Msb, config: MicrosandboxConfig) -> Result<Self, InvalidMicrosandboxConfig> {
        config.validate()?;
        Ok(Self { msb, config })
    }

    /// The sandbox's inspection, or `None` if it does not exist.
    pub(crate) async fn inspect(
        &self,
        name: &str,
    ) -> Result<Option<SandboxInspection>, MsbFailure> {
        let arguments = vec![
            "inspect".into(),
            "--format".into(),
            "json".into(),
            name.into(),
        ];
        match self.msb.run(arguments, COMMAND_TIMEOUT).await {
            Ok(document) => serde_json::from_slice(&document)
                .map(Some)
                .map_err(|error| {
                    MsbFailure::Failed(format!(
                        "msb inspect {name} returned unreadable JSON: {error}"
                    ))
                }),
            Err(MsbFailure::SandboxMissing) => Ok(None),
            Err(failure) => Err(failure),
        }
    }

    /// Fails closed: a sandbox whose mounts and network policy cannot be
    /// confirmed is never used.
    pub(crate) fn confirm_isolation(
        &self,
        name: &str,
        inspection: &SandboxInspection,
    ) -> Result<(), String> {
        inspection
            .verify_mounts(&self.config.workspace)
            .and_then(|()| inspection.verify_network(&self.config.egress))
            .map_err(|problem| format!("sandbox {name} {problem}"))
    }

    /// Lifecycle commands run without `--quiet`, which also hides `msb rm`'s
    /// and `msb start`'s report that a sandbox does not exist.
    async fn lifecycle(&self, command: &str, name: &str) -> Result<(), SandboxProviderError> {
        let arguments = vec![command.into(), name.into()];
        self.msb
            .run(arguments, COMMAND_TIMEOUT)
            .await
            .map(drop)
            .map_err(|failure| lifecycle_error(name, failure))
    }

    async fn remove(&self, name: &str) -> Result<(), MsbFailure> {
        let arguments: Vec<OsString> = vec!["rm".into(), "--force".into(), name.into()];
        self.msb.run(arguments, COMMAND_TIMEOUT).await.map(drop)
    }

    /// Accepts a sandbox only if this request created it, its network policy
    /// is the configured allowlist and it mounts this provider's workspace.
    async fn adopt(
        &self,
        name: &str,
        request_id: Uuid,
        provenance: Provenance,
    ) -> Result<String, SandboxProviderError> {
        let inspection = self
            .inspect(name)
            .await
            .map_err(|failure| lifecycle_error(name, failure))?
            .ok_or_else(|| {
                SandboxProviderError::Retryable(format!("sandbox {name} was not created"))
            })?;
        let request = request_id.to_string();
        if inspection.label(REQUEST_LABEL) != Some(request.as_str()) {
            return Err(SandboxProviderError::Rejected(format!(
                "sandbox {name} exists but was not created for request {request}"
            )));
        }
        // A sandbox whose network Mirror cannot vouch for must not linger with
        // a workspace mounted in it, whoever made it, so this check comes
        // first.
        if let Err(problem) = inspection.verify_network(&self.config.egress) {
            return Err(self
                .discard(name, format!("sandbox {name} {problem}"))
                .await);
        }
        if let Err(problem) = inspection.verify_mounts(&self.config.workspace) {
            let problem = format!("sandbox {name} {problem}");
            return Err(match provenance {
                Provenance::Existing => SandboxProviderError::Rejected(problem),
                Provenance::New => self.discard(name, problem).await,
            });
        }
        Ok(name.to_string())
    }

    /// Removes a sandbox that cannot be used, so a retry can create it again.
    async fn discard(&self, name: &str, problem: String) -> SandboxProviderError {
        SandboxProviderError::Rejected(match self.remove(name).await {
            Ok(()) | Err(MsbFailure::SandboxMissing) => format!("{problem}; it was removed"),
            Err(failure) => format!("{problem}; removing it failed: {failure}"),
        })
    }

    /// Polls until `settled` returns a result or `timeout` passes.
    async fn poll<T>(
        &self,
        name: &str,
        timeout: Duration,
        mut settled: impl FnMut(Option<&SandboxInspection>) -> PollStep<T>,
    ) -> Result<T, SandboxProviderError> {
        let deadline = Instant::now() + timeout;
        loop {
            let inspection = self
                .inspect(name)
                .await
                .map_err(|failure| lifecycle_error(name, failure))?;
            if let Some(result) = settled(inspection.as_ref()) {
                return result;
            }
            if Instant::now() >= deadline {
                let status = inspection.map(|inspection| inspection.status);
                return Err(SandboxProviderError::Retryable(format!(
                    "sandbox {name} was still {status:?} after {timeout:?}"
                )));
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }
}

#[async_trait]
impl SandboxProvider for MicrosandboxProvider {
    async fn create(
        &self,
        request_id: Uuid,
        session_id: Uuid,
    ) -> Result<String, SandboxProviderError> {
        let name = sandbox_name(request_id);
        let existing = self
            .inspect(&name)
            .await
            .map_err(|failure| lifecycle_error(&name, failure))?;
        if existing.is_some() {
            return self.adopt(&name, request_id, Provenance::Existing).await;
        }
        let arguments = self.config.create_arguments(&name, request_id, session_id);
        let provenance = match self.msb.run(arguments, CREATE_TIMEOUT).await {
            Ok(_) => Provenance::New,
            // Another create for the same request took the name first.
            Err(MsbFailure::SandboxExists) => Provenance::Existing,
            // A failed or timed-out create may still have made the sandbox,
            // so only one that still does not exist is a failure.
            Err(failure) => {
                if self.inspect(&name).await.ok().flatten().is_none() {
                    return Err(lifecycle_error(&name, failure));
                }
                Provenance::New
            }
        };
        self.adopt(&name, request_id, provenance).await
    }

    async fn wait_until_ready(&self, name: &str) -> Result<(), SandboxProviderError> {
        self.poll(name, READY_TIMEOUT, |inspection| {
            let inspection = match inspection {
                None => return Some(Err(gone(name))),
                Some(inspection) => inspection,
            };
            match inspection.status {
                SandboxStatus::Running => Some(
                    self.confirm_isolation(name, inspection)
                        .map_err(SandboxProviderError::Rejected),
                ),
                SandboxStatus::Crashed => Some(Err(SandboxProviderError::Rejected(format!(
                    "sandbox {name} crashed while starting"
                )))),
                _ => None,
            }
        })
        .await
    }

    async fn liveness(&self, name: &str) -> Result<SandboxLiveness, SandboxProviderError> {
        let inspection = self
            .inspect(name)
            .await
            .map_err(|failure| lifecycle_error(name, failure))?;
        let Some(inspection) = inspection else {
            return Ok(SandboxLiveness::Gone);
        };
        match inspection.status {
            SandboxStatus::Running => Ok(SandboxLiveness::Ready),
            SandboxStatus::Stopped | SandboxStatus::Crashed | SandboxStatus::Paused => {
                Ok(SandboxLiveness::Suspended)
            }
            status => Err(SandboxProviderError::Retryable(format!(
                "sandbox {name} is {status:?}"
            ))),
        }
    }

    async fn suspend(&self, name: &str) -> Result<(), SandboxProviderError> {
        self.lifecycle("stop", name).await
    }

    async fn wait_until_suspended(&self, name: &str) -> Result<(), SandboxProviderError> {
        self.poll(name, SUSPEND_TIMEOUT, |inspection| match inspection {
            None => Some(Err(gone(name))),
            Some(inspection) => matches!(
                inspection.status,
                SandboxStatus::Stopped | SandboxStatus::Crashed
            )
            .then_some(Ok(())),
        })
        .await
    }

    async fn resume(&self, name: &str) -> Result<(), SandboxProviderError> {
        let inspection = self
            .inspect(name)
            .await
            .map_err(|failure| lifecycle_error(name, failure))?
            .ok_or_else(|| gone(name))?;
        match inspection.status {
            SandboxStatus::Running | SandboxStatus::Starting => Ok(()),
            SandboxStatus::Paused => self.lifecycle("resume", name).await,
            SandboxStatus::Stopped | SandboxStatus::Crashed => self.lifecycle("start", name).await,
            status => Err(SandboxProviderError::Retryable(format!(
                "sandbox {name} is {status:?} and cannot be resumed yet"
            ))),
        }
    }

    async fn delete(&self, name: &str) -> Result<(), SandboxProviderError> {
        match self.remove(name).await {
            Ok(()) | Err(MsbFailure::SandboxMissing) => Ok(()),
            Err(failure) => Err(lifecycle_error(name, failure)),
        }
    }
}

/// The VM for a provisioning request. One request has one VM, so retrying a
/// request finds the VM it already created.
pub fn sandbox_name(request_id: Uuid) -> String {
    format!("mirror-ws-{}", request_id.simple())
}

fn gone(name: &str) -> SandboxProviderError {
    SandboxProviderError::Rejected(format!("sandbox {name} does not exist"))
}

fn lifecycle_error(name: &str, failure: MsbFailure) -> SandboxProviderError {
    match failure {
        MsbFailure::SandboxMissing => gone(name),
        MsbFailure::SandboxExists => {
            SandboxProviderError::Retryable(format!("sandbox {name} already exists"))
        }
        MsbFailure::TimedOut(detail) | MsbFailure::Failed(detail) => {
            SandboxProviderError::Retryable(detail)
        }
    }
}

#[cfg(test)]
#[path = "microsandbox_provider_test.rs"]
mod tests;
