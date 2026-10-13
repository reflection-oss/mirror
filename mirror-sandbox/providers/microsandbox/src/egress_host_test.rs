use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

#[test]
fn parse_lowercases_and_keeps_a_subdomain_wildcard() {
    let host = EgressHost::parse("*.GitHubUserContent.com").expect("valid host");

    assert_eq!(host.name(), "githubusercontent.com");
    assert!(host.includes_subdomains());
    assert_eq!(host.to_string(), "*.githubusercontent.com");
}

#[test]
fn parse_rejects_text_that_would_add_a_network_rule() {
    assert_matches!(
        EgressHost::parse("example.com,allow@public"),
        Err(InvalidEgressHost(_))
    );
}

#[test]
fn parse_rejects_a_wildcard_over_a_top_level_domain() {
    assert_matches!(EgressHost::parse("*.com"), Err(InvalidEgressHost(_)));
}
