//! Native repository discovery results.

use super::Checkout;

/// VCS metadata locates checkouts but never assigns their conversation identities.
#[derive(Debug)]
pub struct Repository {
    pub key: String,
    pub current: Option<Checkout>,
    pub checkouts: Vec<Checkout>,
}
