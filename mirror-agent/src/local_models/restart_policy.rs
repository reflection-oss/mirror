//! When to bring back a server that exited on its own, and when to stop trying.

use std::time::{Duration, Instant};

/// The streak of restarts since the server last ran steadily.
#[derive(Debug, Default)]
pub struct RestartPolicy {
    /// Restarts scheduled in this streak, each waiting the next `BACKOFF` delay.
    restarts: usize,
    last_error: String,
    ready_since: Option<Instant>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    RetryAfter(Duration),
    /// Stop restarting; the text says why.
    GiveUp(String),
}

/// Waits before each restart in a streak; a failure after the last gives up.
const BACKOFF: [Duration; 4] = [
    Duration::from_secs(1),
    Duration::from_secs(5),
    Duration::from_secs(15),
    Duration::from_secs(60),
];

/// A server that stayed up this long starts a fresh streak when it next exits.
const STREAK_RESET: Duration = Duration::from_secs(300);

impl RestartPolicy {
    /// A server became ready, so no earlier attempt's error explains its next exit.
    pub fn on_ready(&mut self, now: Instant) {
        self.ready_since = Some(now);
        self.last_error.clear();
    }

    /// The server exited without being stopped.
    ///
    /// One that dies soon after it came up continues the streak, like a failed restart.
    pub fn on_exit(&mut self, now: Instant) -> Decision {
        if self
            .ready_since
            .take()
            .is_some_and(|since| now.duration_since(since) > STREAK_RESET)
        {
            self.restarts = 0;
        }
        self.schedule()
    }

    /// A restart attempt could not bring the server back.
    pub fn on_attempt_failed(&mut self, error: String) -> Decision {
        self.last_error = error;
        self.schedule()
    }

    /// The user took control, so earlier failures do not count against the next exit.
    pub fn reset(&mut self) {
        self.restarts = 0;
        self.last_error.clear();
    }

    fn schedule(&mut self) -> Decision {
        let Some(delay) = BACKOFF.get(self.restarts) else {
            let notice = if self.last_error.is_empty() {
                "llama.cpp keeps exiting; stopped restarting it".to_owned()
            } else {
                format!("llama.cpp would not restart: {}", self.last_error)
            };
            self.reset();
            return Decision::GiveUp(notice);
        };
        self.restarts += 1;
        Decision::RetryAfter(*delay)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn restarts_back_off_then_give_up_naming_the_last_error_and_start_over() {
        let mut policy = RestartPolicy::default();
        let now = Instant::now();

        let decisions = [
            policy.on_exit(now),
            policy.on_attempt_failed("port in use".into()),
            policy.on_attempt_failed("port in use".into()),
            policy.on_attempt_failed("port in use".into()),
            policy.on_attempt_failed("port in use".into()),
            policy.on_exit(now),
        ];

        assert_eq!(
            decisions,
            [
                Decision::RetryAfter(BACKOFF[0]),
                Decision::RetryAfter(BACKOFF[1]),
                Decision::RetryAfter(BACKOFF[2]),
                Decision::RetryAfter(BACKOFF[3]),
                Decision::GiveUp("llama.cpp would not restart: port in use".into()),
                Decision::RetryAfter(BACKOFF[0]),
            ]
        );
    }

    #[test]
    fn a_server_that_stayed_up_forgives_earlier_failures() {
        let mut policy = RestartPolicy::default();
        let start = Instant::now();
        policy.on_attempt_failed("flaky".into());
        policy.on_ready(start);

        let decision = policy.on_exit(start + STREAK_RESET + Duration::from_secs(1));

        assert_matches!(decision, Decision::RetryAfter(delay) if delay == BACKOFF[0]);
    }

    #[test]
    fn a_server_that_dies_soon_after_each_start_backs_off_from_one_second() {
        let mut policy = RestartPolicy::default();
        let start = Instant::now();
        let soon = start + Duration::from_secs(2);

        let mut crash_soon_after_start = || {
            policy.on_ready(start);
            policy.on_exit(soon)
        };

        let decisions = [
            crash_soon_after_start(),
            crash_soon_after_start(),
            crash_soon_after_start(),
            crash_soon_after_start(),
            crash_soon_after_start(),
        ];

        assert_eq!(
            decisions,
            [
                Decision::RetryAfter(BACKOFF[0]),
                Decision::RetryAfter(BACKOFF[1]),
                Decision::RetryAfter(BACKOFF[2]),
                Decision::RetryAfter(BACKOFF[3]),
                Decision::GiveUp("llama.cpp keeps exiting; stopped restarting it".into()),
            ]
        );
    }
}
