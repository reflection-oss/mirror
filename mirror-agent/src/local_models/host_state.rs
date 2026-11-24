//! What every reader of the managed server sees: its published state and liveness.

use mirror_protocol::local_models::{
    ExecuteResponse, LocalModelEntry, LocalModelServerPhase, LocalModelState,
};
use tokio::sync::watch;

use super::LocalModelError;
use crate::providers::{LlamaCpp, ModelEvent};

#[derive(Clone, Debug)]
pub struct HostState {
    pub state: LocalModelState,
    pub endpoint: String,
    pub alive: watch::Receiver<bool>,
}

/// Who is asking: the daemon that owns the server, or one that only attached to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Owner,
    Peer,
}

/// The server an operation addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServerRef<'a> {
    /// The shared engine, whichever run is current.
    Managed,
    /// The shared engine as this daemon run handed it out in `LocalModelState.managed_reference`;
    /// it stays valid across restarts of the engine and ends with the daemon.
    Issued(&'a str),
    /// A server the user named by address.
    External(&'a str),
}

/// The `base_url` that selects the daemon's managed server.
pub const MANAGED: &str = "managed";

/// Scheme of a reference to the managed server, bound to the daemon run that handed it out.
pub const MANAGED_SCHEME: &str = "managed://";

impl HostState {
    fn running(&self) -> bool {
        self.alive.has_changed().is_ok() && *self.alive.borrow()
    }

    pub fn view(&self, access: Access) -> LocalModelState {
        let mut state = self.state.clone();
        state.owned = access == Access::Owner;
        if !self.running() {
            // A start or restart in progress is not a stop; any other server that is not
            // running is.
            if ![
                LocalModelServerPhase::Starting,
                LocalModelServerPhase::Restarting,
            ]
            .map(i32::from)
            .contains(&state.phase)
            {
                state.phase = LocalModelServerPhase::Stopped.into();
            }
            state.active_model.clear();
            state.asleep = false;
        }
        state
    }

    /// A client for the server `reference` names.
    pub fn server(&self, reference: ServerRef) -> Result<LlamaCpp, LocalModelError> {
        let endpoint = match reference {
            ServerRef::External(address) => address,
            managed => self.resolve(managed)?,
        };
        Ok(LlamaCpp::new(endpoint, None)?)
    }

    /// The managed engine's endpoint, if it is up and `reference` was handed out by this daemon.
    pub fn resolve(&self, reference: ServerRef) -> Result<&str, LocalModelError> {
        if !self.running() || self.endpoint.is_empty() {
            return Err(LocalModelError::NotRunning);
        }
        self.check_issued(reference)?;
        Ok(&self.endpoint)
    }

    /// Refuse a reference to the managed server that an earlier daemon run handed out.
    pub fn check_issued(&self, reference: ServerRef) -> Result<(), LocalModelError> {
        match reference {
            ServerRef::Issued(issued) if issued != self.state.managed_reference => {
                Err(LocalModelError::StaleReference)
            }
            _ => Ok(()),
        }
    }

    /// The state, and the raw endpoint for a caller holding the current run's reference.
    pub fn status(&self, reference: ServerRef, access: Access) -> ExecuteResponse {
        let endpoint = match reference {
            ServerRef::Issued(_) => self.resolve(reference),
            ServerRef::Managed | ServerRef::External(_) => Ok(""),
        };
        let (endpoint, error) = match endpoint {
            Ok(endpoint) => (endpoint.to_owned(), String::new()),
            Err(error) => (String::new(), error.to_string()),
        };
        ExecuteResponse {
            state: Some(self.view(access)),
            endpoint,
            error,
            ..Default::default()
        }
    }

    /// Follow whether the active model is asleep or was unloaded; true when the state changed.
    ///
    /// The router reports this however the load began, so it is not derived from commands.
    pub fn apply_model_event(&mut self, event: &ModelEvent) -> bool {
        if self.state.active_model != event.model {
            return false;
        }
        let asleep = match event.status.as_str() {
            "sleeping" => true,
            "loaded" if self.state.phase == i32::from(LocalModelServerPhase::Loading) => {
                self.set_phase(LocalModelServerPhase::Serving, &event.model);
                return true;
            }
            "loaded" => false,
            // A load that fails ends unloaded, however it began.
            "unloaded"
                if [
                    LocalModelServerPhase::Loading,
                    LocalModelServerPhase::Serving,
                ]
                .map(i32::from)
                .contains(&self.state.phase) =>
            {
                self.set_phase(LocalModelServerPhase::Ready, "");
                return true;
            }
            // A waking model stays asleep until it is loaded again.
            _ => return false,
        };
        if self.state.asleep == asleep {
            return false;
        }
        self.state.asleep = asleep;
        self.state.revision += 1;
        true
    }

    /// Publish what the router says it holds, after a load that did not complete.
    pub fn settle(&mut self, models: &[LocalModelEntry]) {
        let resident = models
            .iter()
            .find(|model| matches!(model.status.as_str(), "loaded" | "sleeping"));
        let loading = models.iter().find(|model| model.status == "loading");
        match (resident, loading) {
            (Some(model), _) => {
                self.set_phase(LocalModelServerPhase::Serving, &model.name);
                self.state.asleep = model.status == "sleeping";
            }
            (None, Some(model)) => self.set_phase(LocalModelServerPhase::Loading, &model.name),
            (None, None) => self.set_phase(LocalModelServerPhase::Ready, ""),
        }
    }

    /// Publish the server phase and the model it is loading or serving.
    pub fn set_phase(&mut self, phase: LocalModelServerPhase, model: &str) {
        self.state.phase = phase.into();
        self.state.active_model = model.into();
        self.state.asleep = false;
        self.state.revision += 1;
    }
}

impl<'a> ServerRef<'a> {
    pub fn parse(base: &'a str) -> Self {
        match base {
            "" | MANAGED => Self::Managed,
            reference if reference.starts_with(MANAGED_SCHEME) => Self::Issued(reference),
            address => Self::External(address),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    /// A running server serving `model`; dropping the sender stops it.
    fn serving(model: &str) -> (HostState, watch::Sender<bool>) {
        let (alive, lifetime) = watch::channel(true);
        let mut host = HostState {
            state: LocalModelState {
                managed_reference: "managed://127.0.0.1:1/current".into(),
                ..Default::default()
            },
            endpoint: "http://127.0.0.1:2/v1".into(),
            alive: lifetime,
        };
        host.set_phase(LocalModelServerPhase::Serving, model);
        (host, alive)
    }

    #[test]
    fn a_stopped_server_claims_no_model_but_keeps_its_reference() {
        let (host, alive) = serving("org/m:Q4");

        drop(alive);

        let view = host.view(Access::Owner);
        assert_eq!(view.phase, i32::from(LocalModelServerPhase::Stopped));
        assert_eq!(view.active_model, "");
        assert_eq!(view.managed_reference, "managed://127.0.0.1:1/current");
    }

    #[test]
    fn a_restarting_server_is_not_reported_as_stopped() {
        let (mut host, alive) = serving("org/m:Q4");
        host.set_phase(LocalModelServerPhase::Restarting, "org/m:Q4");
        drop(alive);

        let view = host.view(Access::Owner);

        assert_eq!(view.phase, i32::from(LocalModelServerPhase::Restarting));
    }

    #[test]
    fn only_this_daemons_reference_resolves_to_the_endpoint() {
        let (host, _alive) = serving("org/m:Q4");

        let current = host.status(
            ServerRef::Issued("managed://127.0.0.1:1/current"),
            Access::Peer,
        );
        let stale = host.status(
            ServerRef::Issued("managed://127.0.0.1:1/previous"),
            Access::Peer,
        );
        let unbound = host.status(ServerRef::Managed, Access::Peer);

        assert_eq!(current.endpoint, "http://127.0.0.1:2/v1");
        assert_eq!(current.error, "");
        assert_eq!(stale.endpoint, "");
        assert_eq!(stale.error, LocalModelError::StaleReference.to_string());
        assert_eq!(
            unbound.endpoint, "",
            "the raw endpoint needs the issued reference"
        );
    }

    #[test]
    fn a_reference_survives_a_restart_of_the_server_but_not_while_it_is_down() {
        let (mut host, alive) = serving("org/m:Q4");
        let reference = ServerRef::Issued("managed://127.0.0.1:1/current");

        drop(alive);
        assert_matches!(host.resolve(reference), Err(LocalModelError::NotRunning));

        let (_relaunched, lifetime) = watch::channel(true);
        host.alive = lifetime;
        assert_matches!(host.resolve(reference), Ok("http://127.0.0.1:2/v1"));
    }

    #[test]
    fn a_failed_load_leaves_the_model_the_router_still_holds_published() {
        let (mut host, _alive) = serving("org/a:Q4");
        host.set_phase(LocalModelServerPhase::Loading, "org/b:Q4");
        let entry = |name: &str, status: &str| LocalModelEntry {
            name: name.into(),
            status: status.into(),
            ..Default::default()
        };

        host.settle(&[entry("org/a:Q4", "loaded"), entry("org/b:Q4", "unloaded")]);
        assert_eq!(host.state.active_model, "org/a:Q4");
        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Serving));

        host.set_phase(LocalModelServerPhase::Loading, "org/b:Q4");
        host.settle(&[entry("org/a:Q4", "unloaded"), entry("org/b:Q4", "unloaded")]);
        assert_eq!(host.state.active_model, "");
        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Ready));
    }

    #[test]
    fn a_load_the_router_finishes_on_its_own_ends_serving() {
        let (mut host, _alive) = serving("");
        host.settle(&[LocalModelEntry {
            name: "org/m:Q4".into(),
            status: "loading".into(),
            ..Default::default()
        }]);
        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Loading));

        assert!(host.apply_model_event(&ModelEvent {
            model: "org/m:Q4".into(),
            status: "loaded".into(),
        }));

        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Serving));
        assert_eq!(host.state.active_model, "org/m:Q4");
    }

    #[test]
    fn a_load_the_router_fails_on_its_own_ends_ready() {
        let (mut host, _alive) = serving("");
        host.settle(&[LocalModelEntry {
            name: "org/m:Q8_0".into(),
            status: "loading".into(),
            ..Default::default()
        }]);

        assert!(host.apply_model_event(&ModelEvent {
            model: "org/m:Q8_0".into(),
            status: "unloaded".into(),
        }));

        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Ready));
        assert_eq!(host.state.active_model, "");
    }

    #[test]
    fn sleep_follows_the_active_model_only() {
        let (mut host, _alive) = serving("org/m:Q4");
        let event = |model: &str, status: &str| ModelEvent {
            model: model.into(),
            status: status.into(),
        };

        assert!(!host.apply_model_event(&event("other/m:Q4", "sleeping")));
        assert!(host.apply_model_event(&event("org/m:Q4", "sleeping")));
        assert!(!host.apply_model_event(&event("org/m:Q4", "loading")));
        assert!(host.state.asleep, "a waking model is not serving yet");
        assert!(host.apply_model_event(&event("org/m:Q4", "loaded")));
        assert!(!host.state.asleep);
    }

    #[test]
    fn the_router_unloading_the_active_model_leaves_the_server_ready() {
        let (mut host, _alive) = serving("org/m:Q4");

        assert!(host.apply_model_event(&ModelEvent {
            model: "org/m:Q4".into(),
            status: "unloaded".into(),
        }));

        assert_eq!(host.state.active_model, "");
        assert_eq!(host.state.phase, i32::from(LocalModelServerPhase::Ready));
    }
}
