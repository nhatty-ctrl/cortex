"""
Backbone AI — Report Service
Orchestrates: scrape → ingest → signal → recommend → report
This is the main pipeline called by the /api/reports endpoint.
"""
import uuid
from datetime import datetime
from typing import Optional
from api.models.schemas import Report, ScrapeRequest
from scrapers.bright_data import (
    scrape_news_for_ticker,
    fetch_smart_money,
    fetch_macro_news,
    scrape_linkedin_execs,
)
from rag.chroma_store import ingest_news, clear_ticker
from agents.dual_brain import (
    generate_signal,
    generate_recommendation,
    map_news_to_companies,
    gemini_call,
)

async def build_report(req: ScrapeRequest) -> Report:
    """
    Full pipeline:
    1. Scrape news + filings + LinkedIn + macro
    2. Ingest into ChromaDB
    3. Run signal generation (DeepSeek CoT)
    4. Map news impact to companies
    5. Generate strategy recommendation (Gemini Pro)
    6. Compose and return full Report
    """

    # ── Step 1: Scrape ────────────────────────────────────────────
    news_items   = await scrape_news_for_ticker(req.ticker, req.company, depth=req.depth)
    macro_items  = await fetch_macro_news()
    exec_signals = await scrape_linkedin_execs(req.company)
    smart_money  = await fetch_smart_money(req.ticker)

    all_news = news_items + macro_items

    # ── Step 2: Ingest into RAG ───────────────────────────────────
    clear_ticker(req.ticker)                        # refresh — always latest data
    ingest_news(all_news, ticker=req.ticker)

    # ── Step 3: Build news summary for signal gen ─────────────────
    news_text = "\n\n".join([
        f"[{item.source}] {item.title}\n{item.summary}"
        for item in all_news[:15]
    ])

    exec_text = "\n".join([
        f"- {r.get('title', '')} ({r.get('source', '')})"
        for r in exec_signals[:5]
    ])
    if exec_text:
        news_text += f"\n\n## EXECUTIVE / LINKEDIN SIGNALS:\n{exec_text}"

    # ── Step 4: Generate signal (DeepSeek reasoning) ──────────────
    signal = await generate_signal(req.ticker, req.company, news_text)

    # ── Step 5: Impact mapping (which companies affected) ─────────
    impacts = await map_news_to_companies(news_text[:2000], req.ticker, req.company)

    # ── Step 6: Strategy recommendation (Gemini Pro) ──────────────
    recommendation = await generate_recommendation(
        ticker=req.ticker,
        company=req.company,
        signals=[signal],
        smart_money=smart_money,
    )

    # ── Step 7: Executive summary ─────────────────────────────────
    summary_prompt = f"""
Write a 3-paragraph executive summary for an institutional investor about {req.company} ({req.ticker}).

Signal: {signal.signal_type.upper()} | Confidence: {signal.confidence:.0%}
Recommendation: {recommendation.action.upper()} | Horizon: {recommendation.time_horizon}
Key finding: {signal.headline}
Risk: {recommendation.risk_note}
Macro: {recommendation.macro_context}

Be direct, professional, and cite concrete data points. No fluff.
"""
    executive_summary = gemini_call(summary_prompt, model="pro")

    # ── Step 8: Assemble report ───────────────────────────────────
    return Report(
        id=               str(uuid.uuid4()),
        ticker=           req.ticker,
        company_name=     req.company,
        generated_at=     datetime.utcnow(),
        executive_summary=executive_summary,
        news_digest=      all_news[:10],
        signals=          [signal],
        recommendation=   recommendation,
        smart_money=      {"impacts": [i.dict() for i in impacts]},
        sources=          signal.sources + [item.url for item in all_news[:5]],
    )
