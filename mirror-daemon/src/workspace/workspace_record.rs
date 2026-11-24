//! Private provisioning facts; public identity is always the canonical workspace.

use mirror_protocol::daemon::{
    CreateWorkspaceRequest, SwitchWorkspaceSessionRequest, Workspace, WorkspaceKind, WorkspaceState,
};
use tonic::Status;
use uuid::Uuid;

use super::Checkout;

/// Durable creation intent makes retries independent of later metadata edits.
#[derive(Clone, PartialEq, prost::Message)]
pub struct WorkspaceRecord {
    #[prost(message, required, tag = "1")]
    pub workspace: Workspace,
    #[prost(message, required, tag = "2")]
    pub creation: CreateWorkspaceRequest,
    #[prost(message, optional, tag = "3")]
    pub checkout: Option<Checkout>,
    #[prost(string, optional, tag = "4")]
    pub sandbox_id: Option<String>,
    #[prost(bool, tag = "5")]
    pub discovered: bool,
    // Private launch intent; endpoints remain live authority, never persisted here.
    #[prost(message, optional, tag = "6")]
    pub personal_context: Option<mirror_protocol::PersonalContext>,
}

impl WorkspaceRecord {
    /// All conversation identities retained by this workspace, including its current binding.
    pub fn sessions(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.workspace.session_id.as_str()).chain(
            self.workspace
                .previous_session_ids
                .iter()
                .map(String::as_str),
        )
    }

    /// Validate a proposed binding before stopping its worker or changing persisted metadata.
    pub fn switch_session(&self, request: &SwitchWorkspaceSessionRequest) -> Result<Self, Status> {
        if self.workspace.kind == WorkspaceKind::Remote as i32 {
            return Err(Status::failed_precondition(
                "remote sessions are owned by their server",
            ));
        }
        if !matches!(
            WorkspaceState::try_from(self.workspace.state),
            Ok(WorkspaceState::Available | WorkspaceState::Closed)
        ) {
            return Err(Status::failed_precondition("workspace is not available"));
        }
        if request.expected_session_id != self.workspace.session_id {
            return Err(Status::aborted(
                "workspace session changed; inspect its current binding",
            ));
        }
        let session_id = match &request.session_id {
            Some(id) if self.sessions().any(|saved| saved == id) => id.clone(),
            Some(_) => {
                return Err(Status::not_found(
                    "conversation does not belong to this workspace",
                ))
            }
            None => Uuid::new_v4().simple().to_string(),
        };
        let mut record = self.clone();
        if session_id != record.workspace.session_id {
            record
                .workspace
                .previous_session_ids
                .retain(|id| id != &session_id);
            record
                .workspace
                .previous_session_ids
                .push(std::mem::replace(
                    &mut record.workspace.session_id,
                    session_id,
                ));
        }
        record.workspace.state = WorkspaceState::Available as i32;
        record.workspace.error = None;
        Ok(record)
    }
}
