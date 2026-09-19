from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable


_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
_sums: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_counts: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)


def _key(name: str, labels: dict[str, object] | None) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def increment(name: str, value: float = 1, **labels: object) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def gauge(name: str, value: float, **labels: object) -> None:
    with _lock:
        _gauges[_key(name, labels)] = value


def replace_gauges(
    name: str, samples: Iterable[tuple[float, dict[str, object]]]
) -> None:
    """Replace one complete gauge family so removed database labels do not linger."""
    with _lock:
        stale = [key for key in _gauges if key[0] == name]
        for key in stale:
            del _gauges[key]
        for value, labels in samples:
            _gauges[_key(name, labels)] = value


def observe(name: str, value: float, **labels: object) -> None:
    key = _key(name, labels)
    with _lock:
        _sums[key] += value
        _counts[key] += 1


def _format_labels(labels: Iterable[tuple[str, str]]) -> str:
    items = list(labels)
    if not items:
        return ""
    escaped = [
        f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in items
    ]
    return "{" + ",".join(escaped) + "}"


def render_prometheus() -> str:
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        gauges = dict(_gauges)
        sums = dict(_sums)
        counts = dict(_counts)
    for (name, labels), value in sorted(counters.items()):
        lines.append(f"{name}_total{_format_labels(labels)} {value:g}")
    for (name, labels), value in sorted(gauges.items()):
        lines.append(f"{name}{_format_labels(labels)} {value:g}")
    for (name, labels), value in sorted(sums.items()):
        suffix = _format_labels(labels)
        lines.append(f"{name}_sum{suffix} {value:g}")
        lines.append(f"{name}_count{suffix} {counts[(name, labels)]}")
    return "\n".join(lines) + "\n"


def collect_runtime_metrics() -> None:
    try:
        import psutil

        process = psutil.Process()
        gauge("genexam_process_resident_memory_bytes", process.memory_info().rss)
        gauge("genexam_process_cpu_percent", process.cpu_percent(interval=None))
        gauge("genexam_process_threads", process.num_threads())
    except Exception:
        return
