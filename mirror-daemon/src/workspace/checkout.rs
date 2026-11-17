//! Native VCS facts, never workspace or conversation identities.

/// The provider identity is used only to revalidate managed deletion.
#[derive(Clone, PartialEq, prost::Message)]
pub struct Checkout {
    #[prost(string, tag = "1")]
    pub directory: String,
    #[prost(string, tag = "2")]
    pub id: String,
    #[prost(string, tag = "3")]
    pub provider: String,
    #[prost(string, optional, tag = "4")]
    pub name: Option<String>,
    #[prost(bool, tag = "5")]
    pub managed: bool,
}
