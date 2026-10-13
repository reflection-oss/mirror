//! Caller-supplied request keys merged into a provider's request body.

use mirror_protocol::InferenceConfig;
use serde_json::Map;
use serde_json::Value;

use super::LlmError;

/// Merge the inference config's `extra_body` object into `body`.
///
/// Applied after every field the provider computes, so a key the caller
/// names wins: this is how an endpoint's own parameters, such as
/// `chat_template_kwargs` or an integer `reasoning_effort`, reach it without
/// the provider knowing about them. The value is JSON text on the wire, so an
/// integer is forwarded as an integer rather than as a float.
pub fn apply_extra_body(
    body: &mut Map<String, Value>,
    inference: Option<&InferenceConfig>,
) -> Result<(), LlmError> {
    let Some(raw) = inference
        .and_then(|inference| inference.extra_body.as_deref())
        .filter(|raw| !raw.trim().is_empty())
    else {
        return Ok(());
    };
    let parsed: Value = serde_json::from_str(raw)
        .map_err(|error| LlmError::Request(format!("extra_body is not valid JSON: {error}")))?;
    let Value::Object(extra) = parsed else {
        return Err(LlmError::Request(
            "extra_body must be a JSON object".to_string(),
        ));
    };
    body.extend(extra);
    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn inference(extra_body: Option<&str>) -> InferenceConfig {
        InferenceConfig {
            extra_body: extra_body.map(str::to_string),
            ..Default::default()
        }
    }

    fn body() -> Map<String, Value> {
        Map::from_iter([
            ("model".to_string(), json!("m")),
            ("reasoning_effort".to_string(), json!("medium")),
        ])
    }

    #[test]
    fn extra_body_adds_and_overrides_keys() {
        let mut body = body();
        apply_extra_body(
            &mut body,
            Some(&inference(Some(
                r#"{"reasoning_effort": 64, "chat_template_kwargs": {"enable_thinking": false}}"#,
            ))),
        )
        .expect("object merges");
        assert_eq!(
            Value::Object(body),
            json!({
                "model": "m",
                "reasoning_effort": 64,
                "chat_template_kwargs": {"enable_thinking": false},
            })
        );
    }

    #[test]
    fn extra_body_absent_or_blank_leaves_body_alone() {
        let mut body = body();
        apply_extra_body(&mut body, None).expect("absent config is fine");
        apply_extra_body(&mut body, Some(&inference(None))).expect("unset field is fine");
        apply_extra_body(&mut body, Some(&inference(Some("  ")))).expect("blank is fine");
        assert_eq!(Value::Object(body), Value::Object(self::body()));
    }

    #[test]
    fn extra_body_rejects_non_objects_and_bad_json() {
        let mut body = body();
        assert_matches!(
            apply_extra_body(&mut body, Some(&inference(Some("[1]")))),
            Err(LlmError::Request(message)) if message.contains("JSON object")
        );
        assert_matches!(
            apply_extra_body(&mut body, Some(&inference(Some("{nope")))),
            Err(LlmError::Request(message)) if message.contains("not valid JSON")
        );
    }
}
