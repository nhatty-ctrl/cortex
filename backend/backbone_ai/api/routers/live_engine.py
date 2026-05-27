"""
Cortex AI live engine: scheduler, SSE streams, graph state, and watchlist-backed runtime state.
"""

import asyncio
import json
import time
from contextlib import suppress
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.supervisor import cortex_run
from config.settings import settings
from rag.cortex_rag import clear_ticker, ingest, ingest_news_list
from scrapers.bright_data_client import (
    scrape_linkedin_exec,
    scrape_macro_news,
    scrape_news,
    scrape_sec_filings,
    scrape_smart_money,
    set_emit_hook,
)

router = APIRouter()

_signal_queue: asyncio.Queue = asyncio.Queue()
_scrape_queue: asyncio.Queue = asyncio.Queue()
_scheduler_task: asyncio.Task | None = None
_scheduler_lock = asyncio.Lock()
_last_headlines: dict[str, set[str]] = {}
_latest_reports: dict[str, dict] = {}
_graph_state: dict[str, list[dict]] = {"nodes": [], "edges": []}
_refresh_locks: dict[str, asyncio.Lock] = {}

DEFAULT_WATCHLIST = [
    {"ticker": "AAPL", "company": "Apple Inc."},
    {"ticker": "NVDA", "company": "NVIDIA Corporation"},
    {"ticker": "TSLA", "company": "Tesla Inc."},
    {"ticker": "MSFT", "company": "Microsoft Corporation"},
    {"ticker": "GOLD", "company": "Gold Spot Price"},
    {"ticker": "EURUSD", "company": "Euro / US Dollar"},
]
_watchlist: list[dict[str, str]] = [item.copy() for item in DEFAULT_WATCHLIST]


def _normalize_watch_item(ticker: str, company: str) -> dict[str, str]:
    normalized_ticker = ticker.upper().strip()
    return {"ticker": normalized_ticker, "company": company.strip() or normalized_ticker}


def get_watchlist() -> list[dict[str, str]]:
    return [item.copy() for item in _watchlist]


def get_watch_item(ticker: str) -> dict[str, str] | None:
    normalized = ticker.upper().strip()
    return next((item.copy() for item in _watchlist if item["ticker"] == normalized), None)


def add_watch_item(ticker: str, company: str) -> dict[str, str]:
    item = _normalize_watch_item(ticker, company)
    for existing in _watchlist:
        if existing["ticker"] == item["ticker"]:
            existing["company"] = item["company"]
            return existing.copy()
    _watchlist.append(item)
    return item.copy()


def remove_watch_item(ticker: str) -> bool:
    normalized = ticker.upper().strip()
    original_len = len(_watchlist)
    _watchlist[:] = [item for item in _watchlist if item["ticker"] != normalized]
    return len(_watchlist) != original_len


def get_scheduler_status() -> dict:
    if not settings.ENABLE_SCHEDULER:
        return {"status": "disabled", "reason": "ENABLE_SCHEDULER=false"}
    if not settings.has_scraping_credentials:
        return {"status": "disabled", "reason": "missing Bright Data credentials"}
    is_running = _scheduler_task is not None and not _scheduler_task.done()
    return {
        "status": "running" if is_running else "idle",
        "interval_seconds": settings.SCRAPE_INTERVAL_S,
        "watchlist_size": len(_watchlist),
    }


async def _bd_emit(msg: str, source: str = "", status: str = "running"):
    await _scrape_queue.put(
        {
            "type": "scrape_status",
            "message": msg,
            "source": source,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


set_emit_hook(_bd_emit)


def _has_new(ticker: str, items: list[dict]) -> tuple[bool, set[str]]:
    current = {item.get("title", "") for item in items if item.get("title")}
    previous = _last_headlines.get(ticker, set())
    new_titles = current - previous
    return bool(new_titles), new_titles


def _signal_color(signal: str) -> str:
    return {
        "buy": "#1D9E75",
        "bullish": "#1D9E75",
        "hold": "#EF9F27",
        "neutral": "#888780",
        "watch": "#378ADD",
        "bearish": "#D85A30",
        "avoid": "#E24B4A",
        "sell": "#E24B4A",
    }.get((signal or "").lower(), "#888780")


def _ensure_graph_seeded():
    if _graph_state["nodes"]:
        return
    for item in get_watchlist():
        _graph_state["nodes"].append(
            {
                "id": item["ticker"],
                "label": item["ticker"],
                "company": item["company"],
                "signal": "neutral",
                "confidence": 0.5,
                "size": 30,
                "color": "#888780",
                "updated_at": datetime.utcnow().isoformat(),
            }
        )


def _upsert_node(ticker: str, company: str, result: dict):
    summary = result.get("summary", {})
    signal = summary.get("signal", "neutral")
    conf = summary.get("confidence", 0.5)
    node = {
        "id": ticker,
        "label": ticker,
        "company": company,
        "signal": signal,
        "confidence": conf,
        "size": max(20, int(conf * 60)),
        "color": _signal_color(signal),
        "crash_risk": summary.get("crash_risk", "unknown"),
        "rating": summary.get("rating", "NEUTRAL"),
        "updated_at": datetime.utcnow().isoformat(),
    }
    idx = next((i for i, current in enumerate(_graph_state["nodes"]) if current["id"] == ticker), None)
    if idx is not None:
        _graph_state["nodes"][idx] = node
    else:
        _graph_state["nodes"].append(node)

    event_out = result.get("results", {}).get("event_mapper", {})
    if event_out and not event_out.get("error"):
        for impact in event_out.get("output", {}).get("primary_impacts", [])[:3]:
            peer = impact.get("ticker", "").upper()
            if peer and peer != ticker:
                edge_id = f"{ticker}_{peer}"
                if not any(edge["id"] == edge_id for edge in _graph_state["edges"]):
                    _graph_state["edges"].append(
                        {
                            "id": edge_id,
                            "source": ticker,
                            "target": peer,
                            "strength": impact.get("confidence", 0.5),
                            "direction": impact.get("direction", "neutral"),
                        }
                    )


def _get_refresh_lock(ticker: str) -> asyncio.Lock:
    normalized = ticker.upper().strip()
    lock = _refresh_locks.get(normalized)
    if lock is None:
        lock = asyncio.Lock()
        _refresh_locks[normalized] = lock
    return lock


async def refresh_ticker_now(ticker: str, company: str = "", macro_context: str = "") -> dict:
    normalized = ticker.upper().strip()
    item = get_watch_item(normalized)
    company_name = company.strip() or (item["company"] if item else normalized)

    async with _get_refresh_lock(normalized):
        await _process_ticker(normalized, company_name, macro_context)
        report = _latest_reports.get(normalized)
        if report:
            return report

    raise RuntimeError(f"Unable to refresh live data for {normalized}")


async def refresh_watchlist_now() -> dict:
    watchlist = get_watchlist()
    if not watchlist:
        return {
            "nodes": [],
            "edges": _graph_state["edges"],
            "last_update": datetime.utcnow().isoformat(),
        }

    macro_items: list[dict] = []
    macro_text = ""
    try:
        macro_items = await scrape_macro_news()
        ingest_news_list(macro_items, "MACRO")
        macro_text = "\n".join(
            f"{item.get('title', '')}: {item.get('snippet', '')}" for item in macro_items[:8]
        )
        await _signal_queue.put(
            {
                "type": "macro_update",
                "count": len(macro_items),
                "titles": [item.get("title", "") for item in macro_items[:5]],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as exc:
        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"Macro refresh error: {str(exc)[:80]}",
                "source": "manual-refresh",
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    for item in watchlist:
        await refresh_ticker_now(item["ticker"], item["company"], macro_text)

    _ensure_graph_seeded()
    return {
        "nodes": _graph_state["nodes"],
        "edges": _graph_state["edges"],
        "last_update": datetime.utcnow().isoformat(),
    }


async def _process_ticker(ticker: str, company: str, macro_context: str = ""):
    ticker = ticker.upper().strip()
    try:
        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"Starting {ticker} cycle...",
                "source": ticker,
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        news, filings, execs, smart = await asyncio.gather(
            scrape_news(ticker, company, depth="quick"),
            scrape_sec_filings(ticker),
            scrape_linkedin_exec(company),
            scrape_smart_money(ticker),
            return_exceptions=True,
        )
        news = news if isinstance(news, list) else []
        filings = filings if isinstance(filings, list) else []
        execs = execs if isinstance(execs, list) else []
        smart = smart if isinstance(smart, dict) else {}

        has_new, new_titles = _has_new(ticker, news)
        if not has_new and not filings:
            await _scrape_queue.put(
                {
                    "type": "scrape_status",
                    "message": f"No new content for {ticker}; skipping",
                    "source": ticker,
                    "status": "done",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return

        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"{len(new_titles)} new signals for {ticker}; launching agents",
                "source": ticker,
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        clear_ticker(ticker)
        ingest_news_list(news, ticker)
        for filing in filings[:3]:
            ingest(filing.get("content", ""), ticker, "8k", "SEC EDGAR", url=filing.get("url", ""))

        _last_headlines[ticker] = {item.get("title", "") for item in news}

        context = "\n\n".join(
            f"[{item.get('source', '')}] {item.get('title', '')}: {item.get('snippet', '')}".strip()
            for item in news[:12]
        )
        if execs:
            context += "\n\nEXEC SIGNALS:\n" + "\n".join(item.get("title", "") for item in execs[:5])
        if smart.get("whalewisdom"):
            context += f"\n\nSMART MONEY:\n{smart['whalewisdom'][:800]}"
        if macro_context:
            context += f"\n\nMACRO CONTEXT:\n{macro_context[:600]}"

        result = await cortex_run(
            asset=ticker,
            company=company,
            context=context,
            mode="quick",
            signal_q=_signal_queue,
            scrape_q=_scrape_queue,
            headlines=[item.get("title", "") for item in news[:10]],
        )

        _latest_reports[ticker] = result
        _upsert_node(ticker, company, result)
        await _signal_queue.put(
            {
                "type": "graph_update",
                "nodes": _graph_state["nodes"],
                "edges": _graph_state["edges"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as exc:
        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"Error {ticker}: {str(exc)[:80]}",
                "source": ticker,
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


async def scheduler_loop():
    await _scrape_queue.put(
        {
            "type": "scrape_status",
            "message": "Cortex AI scheduler started",
            "source": "system",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    while True:
        watchlist = get_watchlist()
        t0 = time.monotonic()
        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"Cycle start - {len(watchlist)} assets - {datetime.utcnow().strftime('%H:%M UTC')}",
                "source": "scheduler",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        macro_items: list[dict] = []
        try:
            macro_items = await scrape_macro_news()
            ingest_news_list(macro_items, "MACRO")
            await _signal_queue.put(
                {
                    "type": "macro_update",
                    "count": len(macro_items),
                    "titles": [item.get("title", "") for item in macro_items[:5]],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except Exception as exc:
            await _scrape_queue.put(
                {
                    "type": "scrape_status",
                    "message": f"Macro cycle error: {str(exc)[:80]}",
                    "source": "scheduler",
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        macro_text = "\n".join(
            f"{item.get('title', '')}: {item.get('snippet', '')}" for item in macro_items[:8]
        )

        for item in watchlist:
            await _process_ticker(item["ticker"], item["company"], macro_text)
            await asyncio.sleep(3)

        elapsed = time.monotonic() - t0
        sleep_for = max(0, settings.SCRAPE_INTERVAL_S - elapsed)
        await _scrape_queue.put(
            {
                "type": "scrape_status",
                "message": f"Cycle done in {elapsed:.0f}s - next in {sleep_for:.0f}s",
                "source": "scheduler",
                "status": "done",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        await asyncio.sleep(sleep_for)


async def start_scheduler() -> asyncio.Task | None:
    global _scheduler_task
    if not settings.ENABLE_SCHEDULER or not settings.has_scraping_credentials:
        return None
    async with _scheduler_lock:
        if _scheduler_task and not _scheduler_task.done():
            return _scheduler_task
        _scheduler_task = asyncio.create_task(scheduler_loop(), name="cortex-scheduler")
        return _scheduler_task


async def stop_scheduler():
    global _scheduler_task
    async with _scheduler_lock:
        task = _scheduler_task
        _scheduler_task = None
    if not task:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _sse_gen(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=20.0)
            yield f"data: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        except Exception:
            break


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}


@router.get("/stream")
async def signal_stream():
    return StreamingResponse(
        _sse_gen(_signal_queue),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/live-scrape")
async def live_scrape_stream():
    return StreamingResponse(
        _sse_gen(_scrape_queue),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/graph")
async def get_graph():
    if settings.has_scraping_credentials and get_watchlist():
        try:
            return await refresh_watchlist_now()
        except Exception:
            pass

    _ensure_graph_seeded()
    return {
        "nodes": _graph_state["nodes"],
        "edges": _graph_state["edges"],
        "last_update": datetime.utcnow().isoformat(),
        "live": False,
    }


@router.get("/latest/{ticker}")
async def get_latest(ticker: str):
    normalized = ticker.upper().strip()
    if settings.has_scraping_credentials:
        try:
            return await refresh_ticker_now(normalized)
        except Exception:
            pass

    report = _latest_reports.get(normalized)
    if not report:
        return {
            "ticker": normalized,
            "status": "pending",
            "message": "No live report available yet",
        }
    return report


@router.post("/trigger/{ticker}")
async def trigger_now(ticker: str, company: str = ""):
    ticker = ticker.upper().strip()
    asset_info = {
        "AAPL": "Apple Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "MSFT": "Microsoft Corporation",
        "GOLD": "Gold Spot Price",
        "EURUSD": "Euro / US Dollar",
    }
    company_name = company or asset_info.get(ticker, ticker)
    asyncio.create_task(refresh_ticker_now(ticker, company_name))
    return {"status": "triggered", "ticker": ticker, "company": company_name}
