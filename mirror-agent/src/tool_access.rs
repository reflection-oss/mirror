//! Durable, non-model-facing tool access for one conversation.

use std::collections::BTreeSet;

use mirror_protocol::{agent_message, AgentMessage, ExtensionRecord};
use prost_types::{value, ListValue, Struct, Value};

use crate::StoreError;

/// An internal extension snapshot, retained by the existing compaction machinery.
#[derive(Debug, Default)]
pub struct ToolAccess {
    /// Disabled callable tool names.
    pub disabled: BTreeSet<String>,
}

impl ToolAccess {
    /// Reserved identity for worker-owned access snapshots.
    pub const EXTENSION: &str = "mirror.tool-access";

    /// Restore the latest snapshot, including one retained inside a compacted window.
    pub fn restore(mut entries: &[AgentMessage]) -> Result<Self, StoreError> {
        'window: loop {
            for entry in entries.iter().rev() {
                match &entry.payload {
                    Some(agent_message::Payload::Extension(record))
                        if record.extension == Self::EXTENSION =>
                    {
                        let names = record
                            .data
                            .as_ref()
                            .and_then(|data| data.fields.get("disabled"));
                        let Some(Value {
                            kind: Some(value::Kind::ListValue(names)),
                        }) = names
                        else {
                            return Err(StoreError::Persistence(
                                "invalid tool access snapshot".into(),
                            ));
                        };
                        let disabled = names
                            .values
                            .iter()
                            .map(|name| match &name.kind {
                                Some(value::Kind::StringValue(name)) => Ok(name.clone()),
                                _ => Err(StoreError::Persistence(
                                    "invalid disabled tool name".into(),
                                )),
                            })
                            .collect::<Result<_, _>>()?;
                        return Ok(Self { disabled });
                    }
                    Some(agent_message::Payload::ContextWindow(window)) => {
                        if let Some(compaction) = &window.compaction {
                            entries = &compaction.recent;
                            continue 'window;
                        }
                    }
                    _ => {}
                }
            }
            return Ok(Self::default());
        }
    }

    /// Record access without adding a model message or scheduling a turn.
    pub fn record(&self) -> AgentMessage {
        AgentMessage {
            payload: Some(agent_message::Payload::Extension(ExtensionRecord {
                extension: Self::EXTENSION.into(),
                kind: "snapshot".into(),
                data: Some(Struct {
                    fields: [(
                        "disabled".into(),
                        Value {
                            kind: Some(value::Kind::ListValue(ListValue {
                                values: self
                                    .disabled
                                    .iter()
                                    .map(|name| Value {
                                        kind: Some(value::Kind::StringValue(name.clone())),
                                    })
                                    .collect(),
                            })),
                        },
                    )]
                    .into(),
                }),
                survives_compaction: true,
                ..Default::default()
            })),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirror_protocol::{Compaction, ContextWindowRecord};
    use pretty_assertions::assert_eq;

    #[test]
    fn restore_retains_disabled_tools_through_compaction_and_accepts_a_new_snapshot() {
        let access = ToolAccess {
            disabled: ["bash".into(), "mcp__mail__send".into()].into(),
        };
        let window = AgentMessage {
            payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
                compaction: Some(Compaction {
                    recent: vec![access.record()],
                    ..Default::default()
                }),
                ..Default::default()
            })),
            ..Default::default()
        };
        assert_eq!(
            ToolAccess::restore(std::slice::from_ref(&window))
                .expect("restore")
                .disabled,
            access.disabled
        );
        assert!(
            ToolAccess::restore(&[window, ToolAccess::default().record()])
                .expect("new snapshot")
                .disabled
                .is_empty()
        );
    }
}
