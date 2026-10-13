use std::time::Duration;

use pretty_assertions::assert_eq;

use super::{function_call, tool_timeout};

#[test]
fn bash_timeout_includes_the_runner_margin() {
    let timeout = tool_timeout(&function_call(
        "bash",
        r#"{"command":"sleep 1","timeout_seconds":120}"#,
    ));

    assert_eq!(timeout, Duration::from_secs(125));
}

#[test]
fn bash_timeout_cannot_exceed_the_host_limit() {
    let timeout = tool_timeout(&function_call(
        "bash",
        r#"{"command":"sleep infinity","timeout_seconds":1000000000000}"#,
    ));

    assert_eq!(timeout, Duration::from_secs(605));
}

#[test]
fn bash_without_a_timeout_gets_the_native_default() {
    let timeout = tool_timeout(&function_call("bash", r#"{"command":"sleep 60"}"#));

    assert_eq!(timeout, Duration::from_secs(605));
}

#[test]
fn other_tools_use_the_default_timeout() {
    let timeout = tool_timeout(&function_call("read", r#"{"file_path":"README.md"}"#));

    assert_eq!(timeout, Duration::from_secs(30));
}
