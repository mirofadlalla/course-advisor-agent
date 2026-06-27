"""
monitoring/store.py — Thread-safe In-Memory Metrics Store

RESPONSIBILITY:
    Record every /chat request's performance data and expose aggregate
    statistics. No external dependencies — runs entirely in-process.

METRICS TRACKED PER REQUEST:
    - request_id         : UUID for deduplication / drill-down
    - timestamp          : ISO-8601 UTC
    - question_preview   : first 80 chars of user question
    - total_latency_ms   : wall-clock time for the full /chat round-trip
    - agent_process_ms   : time inside chat_service.chat() (agent + tool calls)
    - ttft_ms            : approximated as agent_process_ms (true TTFT needs
                           streaming; we record this proxy and label it clearly)
    - tokens_in          : request tokens from result.usage()
    - tokens_out         : response tokens from result.usage()
    - cost_usd           : computed from Groq llama-3.3-70b pricing
    - success            : True / False
    - error_msg          : None on success, exception message on failure
    - model              : model name string

GROQ PRICING (llama-3.3-70b-versatile, 2025):
    Input:  $0.59 per 1,000,000 tokens
    Output: $0.79 per 1,000,000 tokens

AGGREGATE STATS (computed on-demand):
    - total_requests, success_count, error_count
    - avg / p50 / p95 / p99 latency_ms
    - avg / p95 ttft_ms
    - total_tokens_in, total_tokens_out, total_cost_usd
    - requests_per_minute (rolling 60-second window)
    - system uptime

THREAD SAFETY:
    A single threading.Lock guards all write operations.
    Reads are lock-free (Python list reads are GIL-protected).
"""

import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone


# ─── Groq pricing ─────────────────────────────────────────────────────────────
_COST_PER_1M_IN_USD = 0.59
_COST_PER_1M_OUT_USD = 0.79

# Maximum number of request records to keep in memory
_MAX_RECORDS = 1000


@dataclass
class RequestRecord:
    """Immutable snapshot of a single /chat request's metrics."""
    request_id: str
    timestamp: str           # ISO-8601 UTC
    timestamp_epoch: float   # Unix timestamp for time-range queries
    question_preview: str
    total_latency_ms: float
    agent_process_ms: float
    ttft_ms: float           # proxy for TTFT; equals agent_process_ms
    tokens_in: int
    tokens_out: int
    cost_usd: float
    success: bool
    error_msg: Optional[str]
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestRecord:
    """Record of a RAG index rebuild event."""
    event_id: str
    timestamp: str
    action: str              # "upload" | "rebuild" | "delete"
    filename: Optional[str]
    duration_ms: Optional[float]
    node_count_after: Optional[int]
    success: bool
    error_msg: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsStore:
    """
    Singleton in-memory store for all system metrics.

    Thread-safe via a single Lock for writes. Python's GIL protects
    list/deque reads from concurrent modification.

    Instantiated once at module level — imported as `metrics_store`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: list[RequestRecord] = []
        self._ingest_events: list[IngestRecord] = []

        # Rolling window for req/min calculation
        self._request_timestamps: deque[float] = deque(maxlen=_MAX_RECORDS)

        # Startup timestamp
        self._startup_time: float = time.time()
        self._startup_dt: str = datetime.now(timezone.utc).isoformat()

    # ── Write operations ───────────────────────────────────────────────────────

    def record_request(
        self,
        *,
        question: str,
        total_latency_ms: float,
        agent_process_ms: float,
        tokens_in: int,
        tokens_out: int,
        success: bool,
        error_msg: Optional[str] = None,
        model: str = "groq:llama-3.3-70b-versatile",
    ) -> str:
        """
        Record a completed /chat request.

        Returns the generated request_id for correlation.
        """
        request_id = str(uuid.uuid4())
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        cost_usd = (
            (tokens_in / 1_000_000) * _COST_PER_1M_IN_USD
            + (tokens_out / 1_000_000) * _COST_PER_1M_OUT_USD
        )

        record = RequestRecord(
            request_id=request_id,
            timestamp=now_iso,
            timestamp_epoch=now_epoch,
            question_preview=question[:80],
            total_latency_ms=round(total_latency_ms, 2),
            agent_process_ms=round(agent_process_ms, 2),
            ttft_ms=round(agent_process_ms, 2),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost_usd, 8),
            success=success,
            error_msg=error_msg,
            model=model,
        )

        with self._lock:
            # Keep rolling window bounded
            if len(self._requests) >= _MAX_RECORDS:
                self._requests.pop(0)
            self._requests.append(record)
            self._request_timestamps.append(now_epoch)

        return request_id

    def record_ingest_event(
        self,
        *,
        action: str,
        filename: Optional[str] = None,
        duration_ms: Optional[float] = None,
        node_count_after: Optional[int] = None,
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> str:
        """Record a RAG ingestion event (upload / rebuild / delete)."""
        event_id = str(uuid.uuid4())
        record = IngestRecord(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            filename=filename,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
            node_count_after=node_count_after,
            success=success,
            error_msg=error_msg,
        )
        with self._lock:
            self._ingest_events.append(record)
        return event_id

    # ── Read / aggregate operations ────────────────────────────────────────────

    def get_all_requests(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return paginated request history, newest-first."""
        reversed_list = list(reversed(self._requests))
        return [r.to_dict() for r in reversed_list[offset:offset + limit]]

    def get_summary(self) -> dict:
        """
        Compute and return aggregate statistics.

        Called on every monitor page poll — optimized for speed.
        """
        records = list(self._requests)  # snapshot

        total = len(records)
        success_count = sum(1 for r in records if r.success)
        error_count = total - success_count

        latencies = [r.total_latency_ms for r in records if r.success]
        ttfts = [r.ttft_ms for r in records if r.success]
        agent_times = [r.agent_process_ms for r in records if r.success]

        total_tokens_in = sum(r.tokens_in for r in records)
        total_tokens_out = sum(r.tokens_out for r in records)
        total_cost = sum(r.cost_usd for r in records)

        # Req/min rolling window (last 60 seconds)
        now = time.time()
        recent = sum(1 for ts in self._request_timestamps if now - ts <= 60)

        # Uptime
        uptime_seconds = now - self._startup_time

        def percentile(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = max(0, int(len(sorted_data) * p / 100) - 1)
            return round(sorted_data[idx], 2)

        def safe_mean(data: list[float]) -> float:
            return round(statistics.mean(data), 2) if data else 0.0

        # Recent latency trend (last 20 successful requests)
        recent_records = [r for r in records if r.success][-20:]
        latency_trend = [
            {"ts": r.timestamp, "v": r.total_latency_ms} for r in recent_records
        ]
        ttft_trend = [
            {"ts": r.timestamp, "v": r.ttft_ms} for r in recent_records
        ]

        # Tokens per request over time
        token_trend = [
            {"ts": r.timestamp, "in": r.tokens_in, "out": r.tokens_out}
            for r in recent_records
        ]

        # Req/min per-minute buckets (last 10 minutes)
        rpm_buckets: dict[int, int] = {}
        for ts in list(self._request_timestamps):
            bucket = int(ts // 60)  # floor to minute
            rpm_buckets[bucket] = rpm_buckets.get(bucket, 0) + 1

        sorted_buckets = sorted(rpm_buckets.items())[-10:]
        rpm_history = [
            {
                "minute": datetime.fromtimestamp(b * 60, tz=timezone.utc).strftime("%H:%M"),
                "count": c,
            }
            for b, c in sorted_buckets
        ]

        return {
            "meta": {
                "startup_time": self._startup_dt,
                "uptime_seconds": round(uptime_seconds, 1),
                "uptime_human": _format_uptime(uptime_seconds),
                "max_records": _MAX_RECORDS,
            },
            "requests": {
                "total": total,
                "success": success_count,
                "errors": error_count,
                "error_rate_pct": round((error_count / total * 100) if total else 0, 2),
                "per_minute_rolling": recent,
            },
            "latency_ms": {
                "avg": safe_mean(latencies),
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "min": round(min(latencies), 2) if latencies else 0,
                "max": round(max(latencies), 2) if latencies else 0,
            },
            "ttft_ms": {
                "avg": safe_mean(ttfts),
                "p95": percentile(ttfts, 95),
                "note": "Proxy metric: agent processing time. True streaming TTFT requires SSE mode.",
            },
            "agent_process_ms": {
                "avg": safe_mean(agent_times),
                "p95": percentile(agent_times, 95),
            },
            "tokens": {
                "total_in": total_tokens_in,
                "total_out": total_tokens_out,
                "avg_in": round(total_tokens_in / total, 1) if total else 0,
                "avg_out": round(total_tokens_out / total, 1) if total else 0,
            },
            "cost": {
                "total_usd": round(total_cost, 6),
                "avg_per_request_usd": round(total_cost / total, 8) if total else 0,
                "pricing": {
                    "input_per_1m": _COST_PER_1M_IN_USD,
                    "output_per_1m": _COST_PER_1M_OUT_USD,
                    "model": "llama-3.3-70b-versatile",
                },
            },
            "trends": {
                "latency": latency_trend,
                "ttft": ttft_trend,
                "tokens": token_trend,
                "rpm_history": rpm_history,
            },
            "ingest_events": [e.to_dict() for e in list(self._ingest_events)[-10:]],
        }

    def reset(self) -> None:
        """Clear all stored metrics (admin action)."""
        with self._lock:
            self._requests.clear()
            self._request_timestamps.clear()
            self._ingest_events.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_uptime(seconds: float) -> str:
    """Format seconds as human-readable uptime string."""
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ── Singleton ──────────────────────────────────────────────────────────────────
metrics_store = MetricsStore()
