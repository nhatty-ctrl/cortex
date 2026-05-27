"""Cortex AI — Scrape router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from scrapers.bright_data_client import scrape_news, scrape_sec_filings, scrape_x_accounts, scrape_historical_event
from rag.cortex_rag import retrieve, format_context

router = APIRouter()

class ScrapeReq(BaseModel):
    ticker:  str
    company: str
    depth:   Literal["quick","deep"] = "quick"

@router.post("/news")
async def scrape(req: ScrapeReq):
    try:
        return await scrape_news(req.ticker, req.company, req.depth)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/sec")
async def sec(req: ScrapeReq):
    try:
        return await scrape_sec_filings(req.ticker)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/history")
async def history(query: str):
    """Browse live web for historical market event parallels."""
    try:
        return {"query": query, "result": await scrape_historical_event(query)}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/context")
async def rag_context(ticker: str, query: str):
    """Return RAG context for a ticker — for frontend citation panel."""
    try:
        chunks = retrieve(query, ticker, n=8)
        return {"ticker": ticker, "query": query, "context": format_context(chunks)}
    except Exception as e:
        raise HTTPException(500, str(e))
