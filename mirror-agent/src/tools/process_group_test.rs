use std::os::unix::process::ExitStatusExt;

use assert_matches::assert_matches;
use pretty_assertions::{assert_eq, assert_ne};

use super::*;

#[test]
fn signal_termination_is_not_reported_as_success() {
    let status = std::process::ExitStatus::from_raw(libc::SIGKILL);

    let code = return_code(status);

    assert_matches!(code, 137);
    assert_ne!(code, 0);
    assert_eq!(code, 128 + libc::SIGKILL);
}
