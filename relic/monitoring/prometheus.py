"""Prometheus metrics helpers for RELIC runtime monitoring."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, Tuple

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    # Optional-dependency probe: without prometheus_client the module keeps
    # its own in-process counters and renders the same metric names in the
    # Prometheus text format. Any other import-time error propagates.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = None
    Histogram = None
    generate_latest = None
    _PROM_AVAILABLE = False


if _PROM_AVAILABLE:
    _API_REQUESTS = Counter(
        "relic_api_requests_total",
        "Total API requests.",
        ["method", "path", "status"],
    )
    _API_DURATION = Histogram(
        "relic_api_request_duration_seconds",
        "API request latency in seconds.",
        ["method", "path"],
    )
    _API_AUTH_FAILURES = Counter(
        "relic_api_auth_failures_total",
        "Total API authentication failures.",
        ["path"],
    )
    _API_EXCEPTIONS = Counter(
        "relic_api_exceptions_total",
        "Total API exceptions by type.",
        ["exception_type"],
    )
else:
    _LOCK = Lock()
    _COUNTERS: Dict[Tuple[str, ...], float] = defaultdict(float)
    _DURATION_SUM: Dict[Tuple[str, str], float] = defaultdict(float)
    _DURATION_COUNT: Dict[Tuple[str, str], int] = defaultdict(int)


def record_api_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Record API request count and latency metrics."""
    if _PROM_AVAILABLE:
        _API_REQUESTS.labels(method=method, path=path, status=str(status_code)).inc()
        _API_DURATION.labels(method=method, path=path).observe(max(duration_seconds, 0.0))
        return

    with _LOCK:
        _COUNTERS[("relic_api_requests_total", method, path, str(status_code))] += 1.0
        key = (method, path)
        _DURATION_SUM[key] += max(duration_seconds, 0.0)
        _DURATION_COUNT[key] += 1


def record_api_auth_failure(path: str) -> None:
    """Record API authentication failures."""
    if _PROM_AVAILABLE:
        _API_AUTH_FAILURES.labels(path=path).inc()
        return

    with _LOCK:
        _COUNTERS[("relic_api_auth_failures_total", path)] += 1.0


def record_api_exception(exception_type: str) -> None:
    """Record API exception counts by exception type."""
    if _PROM_AVAILABLE:
        _API_EXCEPTIONS.labels(exception_type=exception_type).inc()
        return

    with _LOCK:
        _COUNTERS[("relic_api_exceptions_total", exception_type)] += 1.0


def _render_fallback_metrics() -> str:
    lines = [
        "# HELP relic_api_requests_total Total API requests.",
        "# TYPE relic_api_requests_total counter",
    ]
    for (metric, method, path, status), value in sorted(_COUNTERS.items()):
        if metric != "relic_api_requests_total":
            continue
        lines.append(
            f'relic_api_requests_total{{method="{method}",path="{path}",status="{status}"}} {value}'
        )

    lines.extend(
        [
            "# HELP relic_api_auth_failures_total Total API authentication failures.",
            "# TYPE relic_api_auth_failures_total counter",
        ]
    )
    for key, value in sorted(_COUNTERS.items()):
        if key[0] != "relic_api_auth_failures_total":
            continue
        lines.append(f'relic_api_auth_failures_total{{path="{key[1]}"}} {value}')

    lines.extend(
        [
            "# HELP relic_api_exceptions_total Total API exceptions by type.",
            "# TYPE relic_api_exceptions_total counter",
        ]
    )
    for key, value in sorted(_COUNTERS.items()):
        if key[0] != "relic_api_exceptions_total":
            continue
        lines.append(f'relic_api_exceptions_total{{exception_type="{key[1]}"}} {value}')

    lines.extend(
        [
            "# HELP relic_api_request_duration_seconds API request latency in seconds.",
            "# TYPE relic_api_request_duration_seconds summary",
        ]
    )
    for (method, path), total in sorted(_DURATION_SUM.items()):
        count = _DURATION_COUNT.get((method, path), 0)
        lines.append(
            f'relic_api_request_duration_seconds_sum{{method="{method}",path="{path}"}} {total}'
        )
        lines.append(
            f'relic_api_request_duration_seconds_count{{method="{method}",path="{path}"}} {count}'
        )

    lines.append("")
    return "\n".join(lines)


def export_prometheus_metrics() -> tuple[str, str]:
    """Export metrics payload and corresponding content type."""
    if _PROM_AVAILABLE:
        return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
    return _render_fallback_metrics(), CONTENT_TYPE_LATEST


__all__ = [
    "export_prometheus_metrics",
    "record_api_auth_failure",
    "record_api_exception",
    "record_api_request",
]
