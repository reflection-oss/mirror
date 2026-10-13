//! Serde adapters that preserve nested protocol values in protobuf wire form.

use mirror_protocol::chat_record;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::InferenceConfig;
use mirror_protocol::Usage;
use prost::Message;
use serde::de::Error as _;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;

pub(crate) mod chat_messages {
    use super::*;

    pub(crate) fn serialize<S>(
        messages: &[chat_record::Message],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        messages
            .iter()
            .map(|message| {
                ChatRecord {
                    message: Some(message.clone()),
                    ..ChatRecord::default()
                }
                .encode_to_vec()
            })
            .collect::<Vec<_>>()
            .serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Vec<chat_record::Message>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<Vec<u8>>::deserialize(deserializer)?
            .into_iter()
            .map(|encoded| {
                ChatRecord::decode(encoded.as_slice())
                    .map_err(D::Error::custom)?
                    .message
                    .ok_or_else(|| D::Error::custom("encoded chat message has no payload"))
            })
            .collect()
    }
}

pub(crate) mod assistant_message {
    use super::*;

    pub(crate) fn serialize<S>(message: &AssistantMessage, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        message.encode_to_vec().serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<AssistantMessage, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = Vec::<u8>::deserialize(deserializer)?;
        AssistantMessage::decode(encoded.as_slice()).map_err(D::Error::custom)
    }
}

pub(crate) mod usage {
    use super::*;

    pub(crate) fn serialize<S>(usage: &Option<Usage>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        usage
            .as_ref()
            .map(Message::encode_to_vec)
            .serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Usage>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Vec<u8>>::deserialize(deserializer)?
            .map(|encoded| Usage::decode(encoded.as_slice()).map_err(D::Error::custom))
            .transpose()
    }
}

pub(crate) mod inference_config {
    use super::*;

    pub(crate) fn serialize<S>(
        inference: &Option<InferenceConfig>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        inference
            .as_ref()
            .map(Message::encode_to_vec)
            .serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Option<InferenceConfig>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<Vec<u8>>::deserialize(deserializer)?
            .map(|encoded| InferenceConfig::decode(encoded.as_slice()).map_err(D::Error::custom))
            .transpose()
    }
}
