use prost::Message;
use std::io;
use std::path::{Component, Path};

#[derive(Clone, PartialEq, Message)]
pub struct RepositoryDefinition {
    #[prost(string, tag = "1")]
    pub url: String,
    #[prost(string, tag = "2")]
    pub revision: String,
    #[prost(string, tag = "3")]
    pub directory: String,
    #[prost(string, repeated, tag = "4")]
    pub eager_skills: Vec<String>,
}

pub fn validate_repository(requested: &RepositoryDefinition) -> io::Result<()> {
    validate_relative(&requested.directory)?;
    if requested.url.len() > 2048
        || requested.eager_skills.len() > 64
        || requested.eager_skills.iter().any(|name| name.len() > 128)
    {
        return Err(invalid(
            "repository request exceeds URL or eager skill limits",
        ));
    }
    let url = url::Url::parse(&requested.url)
        .map_err(|_| invalid("repository requires a credential-free HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || requested.url.contains(['\n', '\r'])
    {
        return Err(invalid("repository requires a credential-free HTTPS URL"));
    }
    let revision = &requested.revision;
    if revision.is_empty()
        || revision.len() > 256
        || revision.starts_with('-')
        || !revision
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/._-".contains(&c))
    {
        return Err(invalid("repository revision must be a commit or ref name"));
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub fn validate_relative(path: &str) -> io::Result<()> {
    if path.len() > 4096
        || path.contains(['\\', '\0'])
        || path.split('/').any(|p| p == ".." || p == ".")
        || Path::new(path)
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        return Err(invalid("expected a root-relative path without traversal"));
    }
    Ok(())
}
