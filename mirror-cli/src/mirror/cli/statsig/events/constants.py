"""Statsig event API limits and the event client's delivery policy."""

import re

STATSIG_ENDPOINT = 'https://events.statsigapi.net/v1/log_event'

# Field limits match the validation in Statsig's Node SDK; metadata is
# measured as stringified JSON.
MAX_EVENT_NAME_LENGTH = 128
MAX_EVENT_VALUE_LENGTH = 128
MAX_METADATA_LENGTH = 4096
# Event names and metadata keys are lowercase identifiers, never user input.
IDENTIFIER_FORMAT = re.compile(r'[a-z][a-z0-9_]*')

# The set of status codes under which we should retry event delivery.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 522, 524, 599})
# Seconds to wait before each retry; about a minute in total, so a brief
# network outage does not lose events.
RETRY_BACKOFF_PATTERN = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# Seconds allowed for one request.
REQUEST_TIMEOUT = 2.0
# Seconds shutdown may wait for in-flight deliveries, so telemetry never
# noticeably delays quitting.
SHUTDOWN_TIMEOUT = 2.0

# maximum number of unsent events to hold
MAX_UNSENT_EVENTS = 500  # this is a decent amount for product analytics
UNSENT_EVENT_MAX_AGE_MS = 5 * 24 * 60 * 60 * 1000
