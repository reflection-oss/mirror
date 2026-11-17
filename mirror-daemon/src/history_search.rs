//! Sequential ownership of local history search and bounded background backfill.

use std::path::PathBuf;
use std::time::Duration;

use mirror_protocol::daemon::{SearchConversationsRequest, SearchConversationsResponse};
use tokio::sync::{mpsc, oneshot, watch};
use tonic::Status;

/// Keep indexing off worker paths and prevent concurrent keystrokes from piling up scans.
#[derive(Clone, Debug)]
pub struct HistorySearch {
    commands: mpsc::Sender<SearchCommand>,
}

type SearchCommand = (
    SearchConversationsRequest,
    oneshot::Sender<Result<SearchConversationsResponse, Status>>,
);

impl HistorySearch {
    pub fn start(database: PathBuf, mut stopping: watch::Receiver<bool>) -> Self {
        let (commands, mut requests) = mpsc::channel::<SearchCommand>(4);
        tokio::spawn(async move {
            let mut refresh = tokio::time::interval(Duration::from_secs(2));
            let mut failed = false;
            loop {
                if *stopping.borrow() {
                    break;
                }
                tokio::select! {
                    _ = stopping.changed() => break,
                    request = requests.recv() => {
                        let Some((request, reply)) = request else { break; };
                        if reply.is_closed() { continue; }
                        if failed {
                            let _ = reply.send(Err(Status::unavailable("Local message indexing is unavailable. History is unchanged.")));
                            continue;
                        }
                        let path = database.clone();
                        let result = tokio::task::spawn_blocking(move || mirror_agent::local::search_conversations(&path, &request))
                            .await.map_err(|_| Status::internal("conversation search stopped"))
                            .and_then(|result| result.map_err(|_| Status::failed_precondition("Local message search is unavailable. Check the daemon version and history index.")));
                        if result.as_ref().is_ok_and(|page| page.indexing) { refresh.reset_immediately(); }
                        let _ = reply.send(result);
                    }
                    _ = refresh.tick() => {
                        let path = database.clone();
                        let batch = tokio::task::spawn_blocking(move || mirror_agent::local::index_conversation_history(&path)).await;
                        let pending = match batch {
                            Ok(Ok(pending)) => { failed = false; pending }
                            _ => { failed = true; false }
                        };
                        refresh.reset_after(if pending { Duration::from_millis(10) } else { Duration::from_secs(2) });
                    }
                }
            }
        });
        Self { commands }
    }

    pub async fn search(
        &self,
        request: SearchConversationsRequest,
    ) -> Result<SearchConversationsResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .try_send((request, reply))
            .map_err(|_| Status::resource_exhausted("Conversation search is busy. Try again."))?;
        result
            .await
            .map_err(|_| Status::unavailable("Conversation search stopped"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use mirror_agent::Store;
    use mirror_protocol::{agent_message, chat_record, AgentMessage, ChatRecord, UserMessage};
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn search_reads_a_live_conversation_without_attaching_to_its_worker() {
        let directory = tempfile::tempdir().expect("fixture");
        let database = directory.path().join("state.db");
        let mut store = mirror_agent::local::SQLiteStore::open(
            &database,
            "session",
            directory.path(),
            true,
            false,
        )
        .expect("store");
        let entry = AgentMessage {
            id: "message".into(),
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                    content: "synthetic albatross".into(),
                    ..Default::default()
                })),
                ..Default::default()
            })),
            ..Default::default()
        };
        store
            .append("session", 0, &mut [entry])
            .await
            .expect("commit");
        assert!(!mirror_agent::local::index_conversation_history(&database).expect("index"));
        let (stop, stopping) = watch::channel(false);
        let search = HistorySearch::start(database, stopping);
        let page = search
            .search(SearchConversationsRequest {
                query: "albatross".into(),
                ..Default::default()
            })
            .await;
        assert_matches!(&page, Ok(page) if page.matches.len() == 1);
        assert_eq!(page.expect("search").matches[0].message_id, "message");
        stop.send(true).expect("stop");
    }
}
