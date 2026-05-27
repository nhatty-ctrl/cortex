"""Report generation endpoints."""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.supervisor import cortex_run
from rag.cortex_rag import clear_ticker, ingest, ingest_news_list
from scrapers.bright_data_client import (
    scrape_linkedin_exec,
    scrape_news,
    scrape_sec_filings,
    scrape_smart_money,
)

router = APIRouter()
_cache: dict[str, dict] = {}


class ReportRequest(BaseModel):
    ticker: str
    company: str
    mode: Literal["quick", "full", "valuation", "insider", "macro"] = "quick"
    portfolio_size: float = 100_000
    entry_price: float = 0.0


@router.post("/generate")
async def generate_report(req: ReportRequest):
    try:
        ticker = req.ticker.upper()
        news, filings, execs, smart = await asyncio.gather(
            scrape_news(ticker, req.company, depth="quick"),
            scrape_sec_filings(ticker),
            scrape_linkedin_exec(req.company),
            scrape_smart_money(ticker),
            return_exceptions=True,
        )
        news = news if isinstance(news, list) else []
        filings = filings if isinstance(filings, list) else []
        execs = execs if isinstance(execs, list) else []
        smart = smart if isinstance(smart, dict) else {}

        clear_ticker(ticker)
        ingest_news_list(news, ticker)
        for filing in filings[:3]:
            ingest(filing.get("content", ""), ticker, "8k", "SEC EDGAR", url=filing.get("url", ""))

        context = "\n\n".join(
            f"[{item.get('source', '')}] {item.get('title', '')}: {item.get('snippet', '')}"
            for item in news[:12]
        )
        if execs:
            context += "\n\nEXEC:\n" + "\n".join(item.get("title", "") for item in execs[:5])
        if smart.get("whalewisdom"):
            context += f"\n\nSMART MONEY:\n{smart['whalewisdom'][:600]}"

        result = await cortex_run(
            asset=ticker,
            company=req.company,
            context=context,
            mode=req.mode,
            portfolio_size=req.portfolio_size,
            entry_price=req.entry_price,
            headlines=[item.get("title", "") for item in news[:10]],
        )
        _cache[ticker] = result
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/{ticker}/latest")
async def latest_report(ticker: str):
    report = _cache.get(ticker.upper())
    if not report:
        raise HTTPException(404, "No report. Call /generate first.")
    return report
