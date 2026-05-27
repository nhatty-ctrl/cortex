"""Cortex AI — Signals router"""
from fastapi import APIRouter, HTTPException
from scrapers.bright_data_client import scrape_news, scrape_macro_news
from rag.cortex_rag import ingest_news_list
from agents.supervisor import cortex_run

router = APIRouter()

@router.get("/macro/latest")
async def macro_latest():
    try:
        return await scrape_macro_news()
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/{ticker}")
async def get_signal(ticker: str, company: str = ""):
    ticker = ticker.upper()
    try:
        news = await scrape_news(ticker, company or ticker, depth="quick")
        ingest_news_list(news, ticker)
        context = "\n".join([f"{n.get('title','')}: {n.get('snippet','')}" for n in news[:10]])
        result  = await cortex_run(asset=ticker, company=company or ticker,
                                    context=context, mode="quick")
        return result.get("summary", {})
    except Exception as e:
        raise HTTPException(500, str(e))
