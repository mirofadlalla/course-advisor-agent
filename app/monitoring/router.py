"""
monitoring/router.py — Monitoring API Endpoints

Endpoints:
    GET  /metrics           — full metrics snapshot (all requests + aggregates)
    GET  /metrics/summary   — lightweight summary for dashboard polling
    GET  /metrics/requests  — paginated request history
    DELETE /metrics         — reset all metrics (admin)
"""

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.monitoring.store import metrics_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["monitoring"])


@router.get("/summary")
def get_metrics_summary():
    """
    Lightweight summary endpoint polled every 5 seconds by the dashboard.
    Returns aggregate stats + recent trends for chart rendering.
    """
    return metrics_store.get_summary()


@router.get("/requests")
def get_request_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    Paginated request history, newest-first.
    """
    records = metrics_store.get_all_requests(limit=limit, offset=offset)
    return {
        "records": records,
        "count": len(records),
        "limit": limit,
        "offset": offset,
    }


@router.get("")
def get_full_metrics():
    """
    Full metrics dump: summary + complete request history.
    Use for debugging or exporting data.
    """
    summary = metrics_store.get_summary()
    all_requests = metrics_store.get_all_requests(limit=1000)
    return {
        "summary": summary,
        "requests": all_requests,
    }


@router.delete("")
def reset_metrics():
    """
    Reset all collected metrics. Admin use only.
    """
    metrics_store.reset()
    logger.info("Metrics store reset via DELETE /metrics")
    return {"status": "ok", "message": "All metrics have been reset."}
