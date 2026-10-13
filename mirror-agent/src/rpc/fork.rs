use crate::StoreError;
use tonic::Status;

pub fn fork_status(error: StoreError) -> Status {
    match error {
        StoreError::InvalidForkTarget { .. } => Status::invalid_argument(error.to_string()),
        StoreError::EntryNotFound { .. } | StoreError::SessionNotFound { .. } => {
            Status::not_found(error.to_string())
        }
        error => Status::unavailable(error.to_string()),
    }
}
