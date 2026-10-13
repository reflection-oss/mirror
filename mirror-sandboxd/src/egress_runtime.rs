use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, watch, OnceCell};
use tokio::time::{timeout_at, Instant};
use tonic::{Status, Streaming};

use crate::egress_connector::EgressConnector;
use mirror_sandbox::{egress_update, EgressEvent, EgressUpdate};

pub(crate) type GuestEgress = Arc<OnceCell<Arc<EgressConnector>>>;
const INITIAL_TIMEOUT: Duration = Duration::from_secs(15);
const REFRESH_WINDOW: Duration = Duration::from_secs(20);

pub(crate) async fn renew(
    mut updates: Streaming<EgressUpdate>,
    events: mpsc::Sender<Result<EgressEvent, Status>>,
    connector: Arc<EgressConnector>,
    mut closing: watch::Receiver<bool>,
) {
    tokio::select! {
        _ = closing.wait_for(|closed| *closed) => {},
        result = exchange(&mut updates, &events, &connector) => {
            if result.is_err() {
                let _ = events.try_send(Err(Status::permission_denied("sandbox egress renewal stopped")));
            }
        }
    }
    connector.revoke();
}

async fn exchange(
    updates: &mut Streaming<EgressUpdate>,
    events: &mpsc::Sender<Result<EgressEvent, Status>>,
    connector: &EgressConnector,
) -> Result<(), ()> {
    let mut expiry = Instant::now() + INITIAL_TIMEOUT;
    for request_id in 1..=u64::MAX {
        timeout_at(
            expiry,
            events.send(Ok(EgressEvent {
                request_id,
                installed: false,
            })),
        )
        .await
        .map_err(|_| ())?
        .map_err(|_| ())?;
        let update = timeout_at(expiry, updates.message())
            .await
            .map_err(|_| ())?
            .map_err(|_| ())?
            .ok_or(())?;
        let Some(egress_update::Update::Identity(identity)) = update.update else {
            return Err(());
        };
        if Instant::now() >= expiry
            || identity.request_id != request_id
            || !connector.install(identity.expires_at, &identity.token)
        {
            return Err(());
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ())?;
        let lifetime = Duration::from_secs(identity.expires_at)
            .checked_sub(now)
            .ok_or(())?;
        expiry = Instant::now() + lifetime;
        timeout_at(
            expiry,
            events.send(Ok(EgressEvent {
                request_id,
                installed: true,
            })),
        )
        .await
        .map_err(|_| ())?
        .map_err(|_| ())?;
        let refresh = expiry - REFRESH_WINDOW.min(lifetime / 2);
        tokio::select! {
            () = tokio::time::sleep_until(refresh) => {},
            () = events.closed() => return Err(()),
            _ = updates.message() => return Err(()),
        }
    }
    Err(())
}
