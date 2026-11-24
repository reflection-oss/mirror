//! What Hugging Face offers as public GGUF, reviewed before consent; never fetch weights here.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use futures_util::future;
use mirror_protocol::local_models::{LocalModelBlob, LocalModelCandidate, LocalModelDownload};
use regex::Regex;
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::Value;

use super::gguf::is_sidecar_name;
use crate::providers::{metadata_client, read_limited, METADATA_LIMIT};

/// Public metadata and reviews, shared by all host clients.
#[derive(Debug)]
pub struct Catalog {
    client: Client,
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("Could not build the Hugging Face client: {0}")]
    Client(&'static str),
    #[error("Cannot reach Hugging Face")]
    Unreachable,
    #[error("Gated Hugging Face repositories are unsupported")]
    Gated,
    #[error("Not found on Hugging Face (private repositories are unsupported)")]
    NotFound,
    #[error("Hugging Face is limiting requests; try again later")]
    RateLimited,
    #[error("Hugging Face refused the request: {0}")]
    Refused(String),
    #[error("Public metadata redirects are unsupported")]
    Redirected,
    #[error("Could not read Hugging Face metadata: {0}")]
    Unreadable(&'static str),
    #[error("Invalid {0}")]
    Invalid(&'static str),
    #[error("{0} is not a repository in the public catalog")]
    NotRepository(String),
    #[error("{0} must be a public, ungated repository with a valid revision")]
    NotPublic(String),
    #[error("Invalid repository filename in {repository}: {name}")]
    InvalidFilename { repository: String, name: String },
    #[error("Repositories with llama.cpp presets are unsupported")]
    Presets,
    #[error("Choose an explicit GGUF quantization")]
    NoQuantization,
    #[error("llama.cpp cannot resolve {0}")]
    Unresolved(String),
    #[error("Quantization {0} is missing or ambiguous")]
    MissingQuantization(String),
    #[error("llama.cpp resolves {quantization} to {file}, not the listed file")]
    ManifestMismatch { quantization: String, file: String },
    #[error("The files of {0} report sizes too large to add up")]
    SizeOverflow(String),
    #[error("{0} must report a size")]
    MissingSize(String),
    #[error("{0} must report a SHA-256 digest")]
    MissingDigest(String),
    #[error("{0} is not UTF-8")]
    LicenseNotText(String),
}

/// Whether featured repositories lead the ranking, as they do for an empty search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Featured {
    First,
    InPlace,
}

#[derive(Debug, Deserialize)]
struct Repository {
    id: String,
    #[serde(default)]
    sha: String,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    siblings: Vec<File>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    gguf: Option<GgufInfo>,
    #[serde(default, rename = "cardData")]
    card_data: Option<CardData>,
}

impl Repository {
    fn is_public(&self) -> bool {
        !self.private && matches!(self.gated, Value::Null | Value::Bool(false))
    }
}

/// What the Hub reads from the GGUF headers.
#[derive(Debug, Deserialize)]
struct GgufInfo {
    /// Total parameter count.
    total: Option<u64>,
    /// The trained context maximum.
    context_length: Option<u64>,
}

/// The licence the publisher declares in the model card, which many repositories
/// carry without a licence file.
#[derive(Debug, Default, Deserialize)]
struct CardData {
    #[serde(default)]
    license: Value,
    license_name: Option<String>,
    license_link: Option<String>,
}

#[derive(Debug, Deserialize)]
struct File {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<Digest>,
}

/// What llama.cpp itself downloads for a tag: the Hub resolves tags and names the weights
/// file. A multimodal repository's projector is also named, and ignored: Mirror is text-only.
#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(rename = "ggufFile")]
    gguf_file: Option<File>,
}

#[derive(Debug, Deserialize)]
struct Digest {
    sha256: String,
}

/// Pipelines that take text in and produce text out, alone or with other modalities.
const TEXT_PIPELINES: [&str; 6] = [
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
    "any-to-any",
];

/// Organizations listed ahead of other results. Membership is a convenience for
/// finding well-known quantizers and labs, not a statement that their files are safe.
const TRUSTED_PUBLISHERS: [&str; 14] = [
    "unsloth",
    "ggml-org",
    "bartowski",
    "lmstudio-community",
    "Qwen",
    "google",
    "meta-llama",
    "microsoft",
    "mistralai",
    "nvidia",
    "openai",
    "deepseek-ai",
    "ibm-granite",
    "HuggingFaceTB",
];

const QUANT: &str = r"(?:IQ[1-6](?:_[A-Z0-9]+)+|Q[2-8](?:_[A-Z0-9]+)+|BF16|F16|F32|MXFP[0-9]+)";

const MANIFEST_LIMIT: usize = 256 * 1024;
const LICENSE_LIMIT: usize = 256 * 1024;
/// Reasons Hugging Face gives for refusing a request are a sentence; anything larger is not one.
const REFUSAL_LIMIT: usize = 4 * 1024;

/// Repositories listed first on an empty search, in this order, whatever their download ranking.
const FEATURED: [&str; 2] = ["unsloth/flint-GGUF", "RecursiveSI/flint-int4-gguf"];

static REPOSITORY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}/[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
        .expect("the repository pattern is a constant that compiles")
});

static LICENSE_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:licen[cs]e|copying)(?:[._-].*)?$")
        .expect("the license file pattern is a constant that compiles")
});

static QUANTIZED_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)(?:^|[-_.])({QUANT})(?:-\d{{5}}-of-\d{{5}})?\.gguf$"
    ))
    .expect("the quantization pattern is built from constants that compile")
});

static SHARD_FILE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(.+)-(\d{5})-of-(\d{5})\.gguf$")
        .expect("the shard pattern is a constant that compiles")
});

impl Catalog {
    pub fn new() -> Result<Self, CatalogError> {
        Ok(Self {
            client: metadata_client().map_err(CatalogError::Client)?,
        })
    }

    /// `agent` is for endpoints that answer differently to llama.cpp than to a browser.
    async fn read(
        &self,
        url: Url,
        limit: usize,
        agent: Option<&str>,
    ) -> Result<Vec<u8>, CatalogError> {
        let mut request = self.client.get(url);
        if let Some(agent) = agent {
            request = request.header(reqwest::header::USER_AGENT, agent);
        }
        let response = request
            .send()
            .await
            .map_err(|_| CatalogError::Unreachable)?;
        if response.status().is_client_error() || response.status().is_server_error() {
            return Err(refusal(response).await);
        }
        // Redirects are not followed, so a redirect arrives here as a success-less status.
        if !response.status().is_success() {
            return Err(CatalogError::Redirected);
        }
        read_limited(response, limit)
            .await
            .map_err(CatalogError::Unreadable)
    }

    /// Search beyond the featured models, without downloading anything.
    ///
    /// Only repositories that accept text and answer in text are returned, trusted
    /// publishers first, after the featured ones on an empty search.
    pub async fn search(&self, query: &str) -> Result<Vec<LocalModelCandidate>, CatalogError> {
        let query = query.trim();
        let mut url = url(&["api", "models"])?;
        url.query_pairs_mut().extend_pairs([
            ("filter", "gguf"),
            ("sort", "downloads"),
            ("direction", "-1"),
            // Filtering drops many hits, so ask for more than the list shows.
            ("limit", "100"),
            ("expand[]", "pipeline_tag"),
            ("expand[]", "tags"),
            ("expand[]", "gguf"),
            ("expand[]", "downloads"),
            ("expand[]", "gated"),
            ("expand[]", "private"),
        ]);
        if !query.is_empty() {
            url.query_pairs_mut()
                .append_pair("search", &query.chars().take(200).collect::<String>());
        }
        // One entry in an unexpected shape is skipped; the rest of the page still lists.
        let mut found: Vec<LocalModelCandidate> =
            serde_json::from_slice::<Vec<Value>>(&self.read(url, METADATA_LIMIT, None).await?)
                .map_err(|_| CatalogError::Invalid("catalog entries"))?
                .into_iter()
                .filter_map(|entry| serde_json::from_value::<Repository>(entry).ok())
                .filter(|entry| {
                    entry.is_public() && is_repository(&entry.id) && accepts_text(entry)
                })
                .map(|entry| candidate(&entry, entry.id.clone(), 0, String::new()))
                .collect();
        if query.is_empty() {
            let missing = FEATURED
                .into_iter()
                .filter(|name| found.iter().all(|item| item.name != *name));
            let featured = future::join_all(missing.map(|name| self.featured(name))).await;
            found.splice(0..0, featured);
        }
        let featured = if query.is_empty() {
            Featured::First
        } else {
            Featured::InPlace
        };
        rank(&mut found, featured);
        Ok(found)
    }

    /// A featured repository's entry, from its own metadata when the Hub answers.
    async fn featured(&self, name: &str) -> LocalModelCandidate {
        match self.details(name).await {
            Ok(data) => candidate(&data, name.into(), 0, String::new()),
            Err(_) => LocalModelCandidate {
                name: name.into(),
                trusted: is_trusted(name),
                ..Default::default()
            },
        }
    }

    async fn manifest(&self, repo: &str, tag: &str) -> Result<File, CatalogError> {
        require_repository(repo)?;
        let manifest: Manifest = serde_json::from_slice(
            &self
                .read(
                    url(&["v2", repo, "manifests", tag])?,
                    MANIFEST_LIMIT,
                    Some("llama-cpp"),
                )
                .await?,
        )
        .map_err(|_| CatalogError::Invalid("llama.cpp manifest"))?;
        manifest
            .gguf_file
            .ok_or_else(|| CatalogError::Unresolved(format!("{repo}:{tag}")))
    }

    async fn details(&self, name: &str) -> Result<Repository, CatalogError> {
        require_repository(name)?;
        let mut url = url(&["api", "models", name])?;
        url.query_pairs_mut().append_pair("blobs", "true");
        let data: Repository = serde_json::from_slice(&self.read(url, METADATA_LIMIT, None).await?)
            .map_err(|_| CatalogError::Invalid("repository metadata"))?;
        if data.id != name || !data.is_public() || !hex(&data.sha, 40) {
            return Err(CatalogError::NotPublic(name.into()));
        }
        check_files(&data)?;
        Ok(data)
    }

    /// List the quantizations of a repository with their download size.
    ///
    /// `memory` is the device memory llama.cpp reports, used to say whether the
    /// weights fit; the exact context is fitted by llama.cpp when the model loads.
    pub async fn variants(
        &self,
        name: &str,
        memory: Option<u64>,
    ) -> Result<Vec<LocalModelCandidate>, CatalogError> {
        let data = self.details(name).await?;
        // Only recognized quantizations are offered: they are the names llama.cpp's router
        // resolves to the files reviewed here.
        variants(&data)
            .into_iter()
            .map(|(quant, files)| {
                let variant = format!("{name}:{quant}");
                let bytes = download_size(&files)
                    .ok_or_else(|| CatalogError::SizeOverflow(variant.clone()))?;
                Ok(candidate(&data, variant, bytes, fit_note(bytes, memory)))
            })
            .collect()
    }

    /// Capture the exact files, revision and full publisher license before consent.
    pub async fn inspect(&self, name: &str) -> Result<LocalModelDownload, CatalogError> {
        let (repo, quant) = name.split_once(':').ok_or(CatalogError::NoQuantization)?;
        let data = self.details(repo).await?;
        let selected = self.manifest(repo, quant).await?;
        let files = selected_files(&data, quant, &selected)?;
        let license = license_notice(
            repo,
            &self.license_text(repo, &data).await?,
            declared_license(&data).as_deref(),
        );
        Ok(LocalModelDownload {
            name: name.into(),
            blobs: blobs(&files)?,
            quantization: quant.into(),
            license,
            revision: data.sha,
        })
    }

    /// The text of the license files at the top of the repository.
    async fn license_text(&self, repo: &str, data: &Repository) -> Result<String, CatalogError> {
        let mut terms = Vec::new();
        for file in &data.siblings {
            if !file.rfilename.contains('/') && LICENSE_FILE.is_match(&file.rfilename) {
                let url = url(&[repo, "raw", &data.sha, &file.rfilename])?;
                let bytes = self.read(url, LICENSE_LIMIT, None).await?;
                terms.push(
                    String::from_utf8(bytes)
                        .map_err(|_| CatalogError::LicenseNotText(file.rfilename.clone()))?,
                );
            }
        }
        Ok(terms.join("\n\n"))
    }
}

/// A Hugging Face URL whose path is `parts`, each split at `/` and percent-encoded.
pub(super) fn url(parts: &[&str]) -> Result<Url, CatalogError> {
    let invalid = || CatalogError::Invalid("metadata URL");
    let mut url = Url::parse("https://huggingface.co/").map_err(|_| invalid())?;
    url.path_segments_mut()
        .map_err(|()| invalid())?
        .clear()
        .extend(parts.iter().flat_map(|part| part.split('/')));
    Ok(url)
}

fn is_repository(name: &str) -> bool {
    REPOSITORY.is_match(name)
}

fn require_repository(name: &str) -> Result<(), CatalogError> {
    if is_repository(name) {
        Ok(())
    } else {
        Err(CatalogError::NotRepository(name.into()))
    }
}

/// The Hub's pipeline tag, else the task tag untagged repositories often carry instead.
fn pipeline(entry: &Repository) -> &str {
    match entry.pipeline_tag.as_deref() {
        Some(tag) => tag,
        None if entry
            .tags
            .iter()
            .any(|tag| matches!(tag.as_str(), "text-generation" | "conversational")) =>
        {
            "text-generation"
        }
        None => "",
    }
}

/// Whether the repository takes text in and answers in text, possibly with other modalities.
fn accepts_text(entry: &Repository) -> bool {
    TEXT_PIPELINES.contains(&pipeline(entry))
}

fn is_trusted(name: &str) -> bool {
    let owner = name.split('/').next().unwrap_or_default();
    TRUSTED_PUBLISHERS
        .iter()
        .any(|trusted| trusted.eq_ignore_ascii_case(owner))
}

fn candidate(
    data: &Repository,
    name: String,
    download_bytes: u64,
    fit: String,
) -> LocalModelCandidate {
    let gguf = data.gguf.as_ref();
    LocalModelCandidate {
        trusted: is_trusted(&data.id),
        pipeline: pipeline(data).to_owned(),
        parameters: gguf.and_then(|gguf| gguf.total).unwrap_or_default(),
        max_context: gguf
            .and_then(|gguf| gguf.context_length)
            .unwrap_or_default(),
        downloads: data.downloads,
        name,
        download_bytes,
        fit,
        ..Default::default()
    }
}

/// Featured repositories first in their listed order when `featured` says so, then trusted
/// publishers.
///
/// The sort is stable, so each group keeps the Hub's download ranking.
fn rank(found: &mut [LocalModelCandidate], featured: Featured) {
    found.sort_by_key(|item| {
        let position = FEATURED
            .iter()
            .position(|name| *name == item.name)
            .filter(|_| featured == Featured::First);
        (position.unwrap_or(FEATURED.len()), !item.trusted)
    });
}

/// Refuse file names that could escape the cache, and repositories that bring their own presets.
fn check_files(data: &Repository) -> Result<(), CatalogError> {
    for file in &data.siblings {
        let name = &file.rfilename;
        if name.split('/').any(|part| matches!(part, "" | "." | ".."))
            || name.contains('\\')
            || name.chars().any(char::is_control)
        {
            return Err(CatalogError::InvalidFilename {
                repository: data.id.clone(),
                name: name.clone(),
            });
        }
        if name == "preset.ini" {
            return Err(CatalogError::Presets);
        }
    }
    Ok(())
}

/// Say whether weights of `bytes` fit in `memory`, leaving room for the context.
fn fit_note(bytes: u64, memory: Option<u64>) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    let Some(memory) = memory else {
        return String::new();
    };
    if bytes > memory {
        return format!(
            "Too large · needs {:.1} of {:.1} GiB",
            bytes as f64 / GIB,
            memory as f64 / GIB
        );
    }
    let spare = memory - bytes;
    let spare_note = format!("{:.1} GiB spare", spare as f64 / GIB);
    if spare < 2 << 30 {
        format!("Tight · {spare_note}")
    } else {
        spare_note
    }
}

/// The license the model card declares, as an identifier with an optional name and link.
fn declared_license(data: &Repository) -> Option<String> {
    let card = data.card_data.as_ref()?;
    let id = match &card.license {
        Value::String(id) => id.clone(),
        Value::Array(ids) => ids
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    };
    let declared = [
        Some(id.as_str()),
        card.license_name.as_deref(),
        card.license_link.as_deref(),
    ]
    .into_iter()
    .flatten()
    .filter(|part| !part.is_empty())
    .collect::<Vec<_>>()
    .join(" · ");
    (!declared.is_empty()).then_some(declared)
}

/// The consent text: license files, else the declared license, else a plain statement
/// that none exists. A missing license is surfaced to the user rather than refused.
fn license_notice(repo: &str, text: &str, declared: Option<&str>) -> String {
    let source = format!("https://huggingface.co/{repo}");
    match (text.trim().is_empty(), declared) {
        (false, Some(declared)) => format!("Declared license: {declared}\n\n{text}"),
        (false, None) => text.to_owned(),
        (true, Some(declared)) => format!(
            "Declared license: {declared}\nNo license file is attached; read the model card at {source}"
        ),
        (true, None) => format!(
            "No license is declared or attached. Without one the publisher grants no rights to use this model; check {source} before continuing."
        ),
    }
}

/// Bytes across `files`; `None` when the sizes the Hub reports do not add up in 64 bits.
fn download_size(files: &[&File]) -> Option<u64> {
    files
        .iter()
        .filter_map(|file| file.size)
        .try_fold(0u64, u64::checked_add)
}

/// The complete quantizations of a repository, each with all of its shards in order.
fn variants(data: &Repository) -> BTreeMap<String, Vec<&File>> {
    let mut groups: BTreeMap<String, Vec<&File>> = BTreeMap::new();
    for file in &data.siblings {
        if is_sidecar_name(&file.rfilename) {
            continue;
        }
        let name = file.rfilename.rsplit('/').next().unwrap_or_default();
        if let Some(captures) = QUANTIZED_FILE.captures(name) {
            groups
                .entry(captures[1].to_ascii_uppercase())
                .or_default()
                .push(file);
        }
    }
    groups.retain(|_, files| {
        files.sort_by_key(|file| &file.rfilename);
        let Some(first) = files.first() else {
            return false;
        };
        match SHARD_FILE.captures(&first.rfilename) {
            Some(parts) => {
                parts[3].parse() == Ok(files.len())
                    && files.iter().enumerate().all(|(index, file)| {
                        file.rfilename
                            == format!("{}-{:05}-of-{:05}.gguf", &parts[1], index + 1, files.len())
                    })
            }
            None => files.len() == 1,
        }
    });
    groups
}

/// The files llama.cpp downloads for `quant`, which must be the file the manifest resolved.
fn selected_files<'a>(
    data: &'a Repository,
    quant: &str,
    selected: &File,
) -> Result<Vec<&'a File>, CatalogError> {
    let group = variants(data)
        .into_iter()
        .find_map(|(name, group)| (name == quant).then_some(group))
        .ok_or_else(|| CatalogError::MissingQuantization(quant.into()))?;
    match group.first() {
        Some(first) if first.rfilename == selected.rfilename => Ok(group),
        _ => Err(CatalogError::ManifestMismatch {
            quantization: quant.into(),
            file: selected.rfilename.clone(),
        }),
    }
}

/// The reviewed files, which must report a size and a SHA-256 digest.
fn blobs(files: &[&File]) -> Result<Vec<LocalModelBlob>, CatalogError> {
    files
        .iter()
        .map(|file| {
            let size = file
                .size
                .filter(|size| *size > 0 && *size < 1 << 63)
                .ok_or_else(|| CatalogError::MissingSize(file.rfilename.clone()))?;
            let digest = file
                .lfs
                .as_ref()
                .filter(|lfs| hex(&lfs.sha256, 64))
                .ok_or_else(|| CatalogError::MissingDigest(file.rfilename.clone()))?;
            Ok(LocalModelBlob {
                path: file.rfilename.clone(),
                digest: digest.sha256.clone(),
                size,
            })
        })
        .collect()
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Why Hugging Face refused a metadata request, in its own words when it gave a reason.
///
/// The Hub answers 401 for a repository that does not exist as well as for a private one, and
/// marks a gated repository's refusals with `X-Error-Code: GatedRepo`.
async fn refusal(response: reqwest::Response) -> CatalogError {
    let status = response.status();
    if response
        .headers()
        .get("x-error-code")
        .is_some_and(|code| code == "GatedRepo")
    {
        return CatalogError::Gated;
    }
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::NOT_FOUND => CatalogError::NotFound,
        StatusCode::TOO_MANY_REQUESTS => CatalogError::RateLimited,
        _ => {
            let reason = read_limited(response, REFUSAL_LIMIT)
                .await
                .ok()
                .and_then(|body| serde_json::from_slice::<Value>(&body).ok())
                .and_then(|reply| reply["error"].as_str().map(str::to_owned));
            CatalogError::Refused(reason.unwrap_or_else(|| status.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    fn repository(pipeline: Option<&str>, tags: &[&str]) -> Repository {
        serde_json::from_value(serde_json::json!({
            "id": "unsloth/model-GGUF",
            "pipeline_tag": pipeline,
            "tags": tags,
        }))
        .expect("repository")
    }

    fn with_files(names: &[&str]) -> Repository {
        serde_json::from_value(serde_json::json!({
            "id": "org/m-GGUF",
            "siblings": names.iter().map(|name| serde_json::json!({"rfilename": name})).collect::<Vec<_>>(),
        }))
        .expect("repository")
    }

    fn quantizations(names: &[&str]) -> Vec<String> {
        variants(&with_files(names)).into_keys().collect()
    }

    #[test]
    fn only_models_that_take_and_answer_in_text_are_listed() {
        assert!(accepts_text(&repository(Some("image-text-to-text"), &[])));
        assert!(!accepts_text(&repository(Some("text-to-image"), &[])));
        assert!(accepts_text(&repository(None, &["gguf", "conversational"])));
        assert!(!accepts_text(&repository(None, &["gguf"])));
    }

    #[test]
    fn quantizations_group_their_shards_and_skip_sidecars() {
        let found = quantizations(&[
            "m-Q4_K_M.gguf",
            "big-Q8_0-00001-of-00002.gguf",
            "big-Q8_0-00002-of-00002.gguf",
            "mmproj-F16.gguf",
            "m-DFlash-Q8_0.gguf",
            "README.md",
        ]);

        assert_eq!(found, ["Q4_K_M", "Q8_0"]);
    }

    #[test]
    fn a_quantization_with_a_missing_shard_or_two_candidates_is_not_offered() {
        assert!(quantizations(&[
            "big-Q8_0-00001-of-00003.gguf",
            "big-Q8_0-00003-of-00003.gguf"
        ])
        .is_empty());
        assert!(quantizations(&["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"]).is_empty());
    }

    #[test]
    fn a_repository_without_a_recognized_quantization_offers_nothing() {
        assert!(quantizations(&["ggml-model-i2_s.gguf", "m-DFlash-Q8_0.gguf"]).is_empty());
    }

    #[test]
    fn shard_sizes_that_overflow_are_rejected() {
        let shard = |size: u64| File {
            rfilename: "m.gguf".into(),
            size: Some(size),
            lfs: None,
        };

        assert_eq!(download_size(&[&shard(2), &shard(3)]), Some(5));
        assert_eq!(download_size(&[&shard(u64::MAX), &shard(1)]), None);
    }

    #[test]
    fn trusted_publishers_match_by_owner_ignoring_case() {
        assert!(is_trusted("Unsloth/anything-GGUF"));
        assert!(!is_trusted("someone/unsloth-GGUF"));
    }

    #[test]
    fn weights_are_judged_against_the_device_memory_llama_cpp_reports() {
        let gib = 1u64 << 30;

        assert_eq!(fit_note(4 * gib, Some(28 * gib)), "24.0 GiB spare");
        assert!(fit_note(27 * gib, Some(28 * gib)).starts_with("Tight"));
        assert!(fit_note(40 * gib, Some(28 * gib)).starts_with("Too large"));
        assert_eq!(fit_note(4 * gib, None), "");
    }

    #[test]
    fn a_license_declared_only_in_the_card_is_shown_not_refused() {
        let data: Repository = serde_json::from_value(serde_json::json!({
            "id": "unsloth/model-GGUF",
            "cardData": {"license": "apache-2.0"},
        }))
        .expect("repository");

        let notice = license_notice("unsloth/model-GGUF", "", declared_license(&data).as_deref());

        assert!(notice.starts_with("Declared license: apache-2.0"));
        assert!(
            license_notice("owner/model-GGUF", "  ", None).starts_with("No license is declared")
        );
    }

    #[test]
    fn featured_repositories_lead_an_empty_search_then_trusted_publishers() {
        let entry = |name: &str, trusted: bool| LocalModelCandidate {
            name: name.into(),
            trusted,
            ..Default::default()
        };
        let mut found = vec![
            entry("other/popular-GGUF", false),
            entry("unsloth/popular-GGUF", true),
            entry("RecursiveSI/flint-int4-gguf", false),
            entry("unsloth/flint-GGUF", true),
        ];

        rank(&mut found, Featured::First);

        let names: Vec<_> = found.iter().map(|item| item.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "unsloth/flint-GGUF",
                "RecursiveSI/flint-int4-gguf",
                "unsloth/popular-GGUF",
                "other/popular-GGUF"
            ]
        );
    }

    #[test]
    fn file_names_that_could_escape_the_cache_and_presets_are_refused() {
        assert_matches!(check_files(&with_files(&["sub/m-Q4_K_M.gguf"])), Ok(()));
        assert_matches!(
            check_files(&with_files(&["../m.gguf"])),
            Err(CatalogError::InvalidFilename { .. })
        );
        assert_matches!(
            check_files(&with_files(&["a\\m.gguf"])),
            Err(CatalogError::InvalidFilename { .. })
        );
        assert_matches!(
            check_files(&with_files(&["m\n.gguf"])),
            Err(CatalogError::InvalidFilename { .. })
        );
        assert_matches!(
            check_files(&with_files(&["preset.ini"])),
            Err(CatalogError::Presets)
        );
    }

    #[test]
    fn a_quantization_resolves_only_to_the_file_the_manifest_names() {
        let data = with_files(&["m-Q4_K_M.gguf"]);
        let manifest = |name: &str| File {
            rfilename: name.into(),
            size: None,
            lfs: None,
        };

        assert_matches!(
            selected_files(&data, "Q4_K_M", &manifest("m-Q4_K_M.gguf")),
            Ok(files) if files.len() == 1
        );
        assert_matches!(
            selected_files(&data, "Q4_K_M", &manifest("evil-Q4_K_M.gguf")),
            Err(CatalogError::ManifestMismatch { .. })
        );
    }

    #[test]
    fn a_reviewed_file_must_report_its_size_and_digest() {
        let file = |size: Option<u64>, sha256: &str| File {
            rfilename: "m.gguf".into(),
            size,
            lfs: Some(Digest {
                sha256: sha256.into(),
            }),
        };
        let digest = "a".repeat(64);

        assert_matches!(blobs(&[&file(Some(1), &digest)]), Ok(_));
        assert_matches!(
            blobs(&[&file(None, &digest)]),
            Err(CatalogError::MissingSize(_))
        );
        assert_matches!(
            blobs(&[&file(Some(1), "not-a-digest")]),
            Err(CatalogError::MissingDigest(_))
        );
    }

    fn reply(status: u16, error_code: &str, body: &'static str) -> reqwest::Response {
        let reply = axum::http::Response::builder()
            .status(status)
            .header("x-error-code", error_code)
            .body(body)
            .expect("response");
        reqwest::Response::from(reply)
    }

    #[tokio::test]
    async fn refusals_tell_gated_from_missing_and_quote_the_hubs_reason() {
        assert_matches!(
            refusal(reply(401, "GatedRepo", "")).await,
            CatalogError::Gated
        );
        assert_matches!(
            refusal(reply(
                401,
                "",
                r#"{"error":"Invalid username or password."}"#
            ))
            .await,
            CatalogError::NotFound
        );
        assert_matches!(
            refusal(reply(400, "", r#"{"error":"Not a valid tag"}"#)).await,
            CatalogError::Refused(reason) if reason == "Not a valid tag"
        );
    }
}
