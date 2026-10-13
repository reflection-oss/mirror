from __future__ import annotations

import asyncio
import collections
import contextlib
import enum
import logging
from typing import Self

import httpx2

from mirror.cli.statsig import config as config_lib
from mirror.cli.statsig import constants as statsig_constants
from mirror.cli.statsig import environment as environment_lib
from mirror.cli.statsig.events import constants, persistence
from mirror.cli.statsig.events import event as event_lib

_logger = logging.getLogger(__name__)


class _Outcome(enum.Enum):
  SENT = enum.auto()
  RETRY = enum.auto()
  DROP = enum.auto()


class EventClient:
  def __init__(self, config: config_lib.StatsigConfig | None = None) -> None:
    """Create a disabled client, permanently disabled without configuration."""
    # config and environment behavior
    self._config = config
    self._environment: environment_lib.Environment | None = None
    # flags
    self._toggling = asyncio.Lock()
    self._shutting_down = asyncio.Event()

    self._http: httpx2.AsyncClient | None = None

    # state
    self._deliveries: set[asyncio.Task[None]] = set()
    self._unsent: collections.deque[bytes] = collections.deque(
      maxlen=constants.MAX_UNSENT_EVENTS
    )

  async def __aenter__(self) -> Self:
    return self

  async def __aexit__(self, *_error: object) -> None:
    await self.disable()

  @property
  def enabled(self) -> bool:
    """Whether events can currently be collected."""
    return self._http is not None

  async def enable(self) -> None:
    """Enable collection if configured; otherwise remain disabled."""
    async with self._toggling:
      if self._config is None or self.enabled:
        return
      self._shutting_down.clear()
      self._http = httpx2.AsyncClient(
        timeout=constants.REQUEST_TIMEOUT,
        follow_redirects=False,
      )
      # on startup, try to resend events that couldn't be delivered last time.
      for encoded in await self._load_saved():
        self._start_delivery(self._http, encoded)

  async def disable(self) -> None:
    async with self._toggling:
      if self._http is None:
        return
      # set to none to prevent new deliveries while closing.
      http = self._http
      self._http = None
      self._shutting_down.set()
      loop = asyncio.get_running_loop()
      deadline = loop.time() + constants.SHUTDOWN_TIMEOUT
      if self._deliveries:
        _, unfinished = await asyncio.wait(
          set(self._deliveries), timeout=constants.SHUTDOWN_TIMEOUT
        )
        for delivery in unfinished:
          delivery.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)
      # save events that haven't been sent yet but have already happened
      # before toggle
      await self._save_unsent()

      with contextlib.suppress(TimeoutError):
        async with asyncio.timeout_at(max(deadline, loop.time())):
          await http.aclose()

  def log_event(self, event: event_lib.Event) -> None:
    """Collect an event only while enabled; disabled calls have no effects."""
    http = self._http
    config = self._config
    if http is None or config is None:
      return
    # create the environment object if it's not already set
    if self._environment is None:
      self._environment = environment_lib.detect()
    try:
      encoded = event_lib.encode(
        event,
        user_id=config.user_id,
        session_id=config.session_id,
        environment=self._environment,
      )
    except (ValueError, TypeError) as error:
      _logger.warning('Statsig event dropped: %s', error)
      return
    self._start_delivery(http, encoded)

  def _start_delivery(self, http: httpx2.AsyncClient, encoded: bytes) -> None:
    delivery = asyncio.create_task(self._deliver_with_retries(http, encoded))
    self._deliveries.add(delivery)
    delivery.add_done_callback(self._deliveries.discard)

  async def _deliver_with_retries(
    self, http: httpx2.AsyncClient, encoded: bytes
  ) -> None:
    delivered = False
    try:
      for delay in (*constants.RETRY_BACKOFF_PATTERN, None):
        outcome = await self._send(http, encoded)
        # if we are told to send or drop then we're good
        if outcome is _Outcome.SENT or outcome is _Outcome.DROP:
          delivered = True
          return
        # if we should stop the backoff
        if delay is None:
          return
        # hold until the delay, or we are closing
        with contextlib.suppress(TimeoutError):
          await asyncio.wait_for(self._shutting_down.wait(), timeout=delay)
        if self._shutting_down.is_set():
          return
    finally:
      if not delivered:
        # if we did not send then we hold in memory for later.
        self._unsent.append(encoded)

  async def _send(self, http: httpx2.AsyncClient, encoded: bytes) -> _Outcome:
    try:
      async with (
        asyncio.timeout(constants.REQUEST_TIMEOUT),
        http.stream(
          'POST',
          constants.STATSIG_ENDPOINT,
          headers={
            'statsig-api-key': statsig_constants.STATSIG_CLIENT_KEY,
            'Content-Type': 'application/json',
          },
          content=encoded,
        ) as response,
      ):
        status = response.status_code
    except (httpx2.TransportError, TimeoutError):
      return _Outcome.RETRY
    except httpx2.HTTPError:
      return _Outcome.DROP
    # Isolation boundary: analytics cannot stop the CLI.
    except Exception as error:  # noqa: BLE001
      _logger.warning('Statsig request failed (%s)', type(error).__name__)
      return _Outcome.DROP

    # if we properly sent the event
    if 200 <= status < 300:
      return _Outcome.SENT

    # if we should attempt a retry
    if status in constants.RETRYABLE_STATUSES:
      return _Outcome.RETRY

    # otherwise just drop
    return _Outcome.DROP

  async def _load_saved(self) -> list[bytes]:
    config = self._config
    if config is None:
      return []
    directory = config.unsent_events_directory
    if directory is None:
      return []
    try:
      return await asyncio.to_thread(persistence.load, directory)
    except OSError as error:
      _logger.warning(
        'Statsig saved events unreadable (%s)', type(error).__name__
      )
      return []

  async def _save_unsent(self) -> None:
    config = self._config
    if config is None:
      return
    unsent = list(self._unsent)
    self._unsent.clear()
    directory = config.unsent_events_directory
    if not unsent or directory is None:
      return
    try:
      await asyncio.to_thread(persistence.save, directory, unsent)
    except OSError as error:
      _logger.warning('Statsig events not saved (%s)', type(error).__name__)


# All CLI workspaces share one launch identity and consent scope.
_instance = EventClient()


def initialize(config: config_lib.StatsigConfig | None) -> None:
  """Install the launch client once at startup, without enabling collection."""
  global _instance
  if _instance.enabled:
    raise RuntimeError('Cannot replace an enabled Statsig client')
  _instance = EventClient(config)


def get() -> EventClient:
  """Return the launch client, permanently disabled if unconfigured."""
  return _instance
