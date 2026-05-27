"""
Cortex AI — Institutional RAG Engine
=====================================
Design principles:
  1. CALCULATION-FIRST: Every claim backed by math shown explicitly
  2. NO pre-stored historical events — HistoricalPlaybook browses live
  3. ChromaDB stores ONLY: scraped filings, news, earnings calls, analyst notes
  4. Multi-strategy retrieval: semantic + metadata filters + recency boost
  5. Finance-aware chunking: SEC section breaks, earnings call speaker splits
  6. Full calculation library: ratios, VaR, Kelly, DCF, Sharpe — all shown

What goes in ChromaDB:
  - SEC filings (8-K, 10-K, 10-Q) scraped every 5 min
  - News articles with full text
  - Earnings call transcripts
  - Analyst report snippets
  - LinkedIn / exec signal text
  - Macro news items
  - Social intelligence posts

What does NOT go in ChromaDB:
  - Historical market events (fetched live by HistoricalPlaybook agent)
  - Pre-baked company fundamentals (always scraped fresh)
"""

import re
import math
import uuid
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from config.settings import settings

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - depends on environment
    chromadb = None
    embedding_functions = None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHROMA SETUP
# ══════════════════════════════════════════════════════════════════════════════

_client = None
_memory_docs: list[dict] = []

def _get_client():
    global _client
    if chromadb is None:
        return None
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client

def _get_collection():
    client = _get_client()
    if client is None:
        return None
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FINANCE-AWARE CHUNKING
# SEC documents split at item boundaries.
# Earnings calls split at speaker changes.
# News split by paragraph with 1-paragraph overlap.
# ══════════════════════════════════════════════════════════════════════════════

SEC_SECTION_RE = re.compile(
    r"(?im)^(ITEM\s+\d+[A-Z]?\b|PART\s+[IVX]+\b|"
    r"Management.{0,10}Discussion|Results of Operations|"
    r"Liquidity and Capital|Risk Factors|"
    r"Consolidated (Balance|Income|Cash))",
    re.MULTILINE
)
SPEAKER_RE = re.compile(r"\n([A-Z][A-Z\s\-\.]+):\s*", re.MULTILINE)


def chunk_document(text: str, doc_type: str = "news", max_size: int = 800) -> list[dict]:
    if doc_type in ("10k", "10q", "8k"):
        return _chunk_sec(text, doc_type, max_size)
    if doc_type == "earnings_call":
        return _chunk_earnings(text)
    return _chunk_paragraphs(text, doc_type, max_size)


def _chunk_sec(text: str, doc_type: str, max_size: int) -> list[dict]:
    splits = SEC_SECTION_RE.split(text)
    headers = SEC_SECTION_RE.findall(text)
    chunks = []
    for i, section in enumerate(splits):
        section = section.strip()
        if len(section) < 80:
            continue
        header = headers[i-1].strip() if i > 0 and i <= len(headers) else "General"
        # Sub-chunk if too long
        if len(section) > max_size * 2:
            paras = [p.strip() for p in section.split("\n\n") if p.strip()]
            window, wlen = [], 0
            for para in paras:
                if wlen + len(para) > max_size and window:
                    chunks.append({"text": "\n\n".join(window), "section": header,
                                   "doc_type": doc_type, "chunk_id": str(uuid.uuid4())})
                    window = window[-1:]
                    wlen   = len(window[0]) if window else 0
                window.append(para)
                wlen += len(para)
            if window:
                chunks.append({"text": "\n\n".join(window), "section": header,
                               "doc_type": doc_type, "chunk_id": str(uuid.uuid4())})
        else:
            chunks.append({"text": f"{header}\n{section}", "section": header,
                           "doc_type": doc_type, "chunk_id": str(uuid.uuid4())})
    return chunks


def _chunk_earnings(text: str) -> list[dict]:
    parts    = SPEAKER_RE.split(text)
    speakers = SPEAKER_RE.findall(text)
    chunks   = []
    for i, content in enumerate(parts):
        content = content.strip()
        if len(content) < 40:
            continue
        speaker   = speakers[i-1].strip() if i > 0 and i <= len(speakers) else "UNKNOWN"
        spk_type  = "management" if any(t in speaker.upper()
                     for t in ["CEO","CFO","COO","PRESIDENT","OPERATOR"]) else "analyst"
        chunks.append({
            "text":         f"{speaker}: {content}",
            "section":      "earnings_call",
            "speaker":      speaker,
            "speaker_type": spk_type,
            "doc_type":     "earnings_call",
            "chunk_id":     str(uuid.uuid4()),
        })
    return chunks


def _chunk_paragraphs(text: str, doc_type: str, max_size: int) -> list[dict]:
    paras  = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 50]
    chunks, window, wlen = [], [], 0
    for para in paras:
        if wlen + len(para) > max_size and window:
            chunks.append({"text": "\n\n".join(window), "section": "body",
                           "doc_type": doc_type, "chunk_id": str(uuid.uuid4())})
            window = window[-1:]
            wlen   = len(window[0]) if window else 0
        window.append(para)
        wlen += len(para)
    if window:
        chunks.append({"text": "\n\n".join(window), "section": "body",
                       "doc_type": doc_type, "chunk_id": str(uuid.uuid4())})
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INGESTION
# ══════════════════════════════════════════════════════════════════════════════

# Red-flag financial phrases (increases retrieval priority)
RED_FLAGS = [
    "going concern", "material weakness", "restatement", "impairment",
    "write-down", "covenant breach", "default", "SEC investigation",
    "class action", "CFO departure", "delayed filing", "auditor change",
    "revenue recognition", "channel stuffing", "margin compression",
]

def ingest(
    text:     str,
    ticker:   str,
    doc_type: str,    # news | 10k | 10q | 8k | earnings_call | analyst | social | macro
    source:   str,
    url:      str = "",
    date:     str = "",
) -> int:
    """Chunk, embed, and store a document. Returns chunks stored."""
    coll   = _get_collection()
    chunks = chunk_document(text, doc_type=doc_type)
    if not chunks:
        return 0

    docs, ids, metas = [], [], []
    for chunk in chunks:
        doc_id = hashlib.md5(f"{ticker}_{url}_{chunk['chunk_id']}".encode()).hexdigest()
        has_rf = int(any(rf in chunk["text"].lower() for rf in RED_FLAGS))
        # Count financial figures in chunk
        fig_count = len(re.findall(r"\$[\d,]+(?:\.\d+)?(?:\s*[BMK])?|\d+(?:\.\d+)?%", chunk["text"]))

        meta = {
            "ticker":      ticker,
            "doc_type":    doc_type,
            "source":      source,
            "url":         url,
            "date":        date or datetime.utcnow().isoformat(),
            "section":     chunk.get("section", "body"),
            "speaker_type":chunk.get("speaker_type", ""),
            "has_red_flag":has_rf,
            "fig_count":   fig_count,
            "chunk_id":    chunk["chunk_id"],
        }
        docs.append(chunk["text"])
        ids.append(doc_id)
        metas.append(meta)

    if coll is not None:
        coll.upsert(documents=docs, ids=ids, metadatas=metas)
    else:
        for doc, doc_id, meta in zip(docs, ids, metas):
            _memory_docs.append({"id": doc_id, "text": doc, "meta": meta})
    return len(docs)


def ingest_news_list(items: list, ticker: str) -> int:
    """Ingest a list of news dicts (from bright_data_client scrapers)."""
    total = 0
    for item in items:
        text = (item.get("title","") + "\n" + item.get("snippet","") +
                "\n" + item.get("full_text",""))
        if len(text.strip()) < 30:
            continue
        total += ingest(
            text=text, ticker=ticker, doc_type="news",
            source=item.get("source","web"), url=item.get("url",""),
            date=item.get("published", datetime.utcnow().isoformat()),
        )
    return total


def clear_ticker(ticker: str):
    """Remove all chunks for a ticker before re-ingesting fresh data."""
    global _memory_docs
    try:
        coll = _get_collection()
        if coll is not None:
            coll.delete(where={"ticker": ticker})
    except Exception:
        pass
    _memory_docs = [doc for doc in _memory_docs if doc["meta"].get("ticker") != ticker]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MULTI-STRATEGY RETRIEVAL
# Semantic search + red-flag boost + recency boost
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query:        str,
    ticker:       str,
    n:            int  = 12,
    boost_redflags: bool = False,
    doc_types:    list  = None,
) -> list[dict]:
    """
    Retrieve top-n most relevant chunks for a query, filtered by ticker.

    Boost logic:
      - Red-flag query ("risk", "danger", "going concern") → extra red-flag chunks
      - Doc-type filter: ["10k", "10q"] for fundamentals, ["news"] for events
    """
    coll  = _get_collection()
    where = {"ticker": ticker}
    if doc_types:
        where["doc_type"] = {"$in": doc_types}

    results = []
    if coll is not None:
        try:
            r = coll.query(query_texts=[query], n_results=n, where=where)
            docs  = r.get("documents",  [[]])[0]
            metas = r.get("metadatas",  [[]])[0]
            dists = r.get("distances",  [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                results.append({"text": doc, "meta": meta,
                                 "score": round(1 - dist, 4)})
        except Exception:
            pass
    else:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        for item in _memory_docs:
            meta = item["meta"]
            if meta.get("ticker") != ticker:
                continue
            if doc_types and meta.get("doc_type") not in doc_types:
                continue
            text_tokens = set(re.findall(r"\w+", item["text"].lower()))
            if not query_tokens:
                score = 0.0
            else:
                score = len(query_tokens & text_tokens) / len(query_tokens)
            if score > 0:
                results.append({"text": item["text"], "meta": meta, "score": round(score, 4)})

    # Red-flag boost: fetch chunks with financial warnings
    if coll is not None and (boost_redflags or any(w in query.lower() for w in ["risk","danger","warning","concern"])):
        try:
            rf_where = {**where, "has_red_flag": 1}
            r2 = coll.query(query_texts=[query], n_results=4, where=rf_where)
            seen = {x["meta"].get("chunk_id") for x in results}
            for doc, meta in zip(r2.get("documents",[[]])[0], r2.get("metadatas",[[]])[0]):
                cid = meta.get("chunk_id")
                if cid not in seen:
                    results.append({"text": doc, "meta": meta, "score": 0.65})
                    seen.add(cid)
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


def format_context(retrieved: list[dict], max_chars: int = 5000) -> str:
    """Format retrieved chunks into LLM-ready context string with source citations."""
    parts, total = [], 0
    for item in retrieved:
        m      = item["meta"]
        header = (f"[{m.get('doc_type','').upper()} | {m.get('source','')} | "
                  f"{m.get('date','')[:10]} | score={item['score']:.2f}]")
        chunk  = f"{header}\n{item['text']}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n━━━\n\n".join(parts) if parts else "No context available."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CALCULATION LIBRARY
# Every calculation shown explicitly — this is the "math shown" layer
# All agents call these functions so math is consistent and auditable
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalcResult:
    name:    str
    value:   float
    formula: str        # human-readable formula used
    inputs:  dict       # the numbers plugged in
    unit:    str = ""
    interpretation: str = ""


def calc_gross_margin(revenue: float, cogs: float) -> CalcResult:
    """Gross Margin = (Revenue - COGS) / Revenue × 100"""
    if revenue == 0:
        return CalcResult("Gross Margin", 0, "N/A", {}, "%", "Cannot calculate — zero revenue")
    val = (revenue - cogs) / revenue * 100
    return CalcResult(
        name="Gross Margin",
        value=round(val, 2),
        formula="(Revenue - COGS) / Revenue × 100",
        inputs={"revenue": revenue, "cogs": cogs},
        unit="%",
        interpretation=("Excellent (>60%)" if val > 60 else
                        "Strong (40-60%)" if val > 40 else
                        "Average (20-40%)" if val > 20 else "Weak (<20%)")
    )


def calc_operating_margin(op_income: float, revenue: float) -> CalcResult:
    """Operating Margin = Operating Income / Revenue × 100"""
    if revenue == 0:
        return CalcResult("Operating Margin", 0, "N/A", {})
    val = op_income / revenue * 100
    return CalcResult(
        name="Operating Margin",
        value=round(val, 2),
        formula="Operating Income / Revenue × 100",
        inputs={"operating_income": op_income, "revenue": revenue},
        unit="%",
        interpretation=("Best-in-class (>25%)" if val > 25 else
                        "Strong (15-25%)" if val > 15 else
                        "Average (5-15%)" if val > 5 else "Weak (<5%)")
    )


def calc_net_margin(net_income: float, revenue: float) -> CalcResult:
    """Net Margin = Net Income / Revenue × 100"""
    if revenue == 0:
        return CalcResult("Net Margin", 0, "N/A", {})
    val = net_income / revenue * 100
    return CalcResult(
        name="Net Margin",
        value=round(val, 2),
        formula="Net Income / Revenue × 100",
        inputs={"net_income": net_income, "revenue": revenue},
        unit="%",
        interpretation=("Excellent (>20%)" if val > 20 else
                        "Good (10-20%)" if val > 10 else
                        "Average (5-10%)" if val > 5 else "Thin (<5%)")
    )


def calc_roic(op_income: float, total_debt: float, equity: float,
              tax_rate: float = 0.21) -> CalcResult:
    """ROIC = NOPAT / Invested Capital = Op.Income×(1-t) / (Debt+Equity)"""
    nopat = op_income * (1 - tax_rate)
    ic    = total_debt + equity
    if ic == 0:
        return CalcResult("ROIC", 0, "N/A", {})
    val = nopat / ic * 100
    return CalcResult(
        name="ROIC",
        value=round(val, 2),
        formula="NOPAT / Invested Capital × 100, where NOPAT = Op.Income × (1 - tax_rate)",
        inputs={"op_income": op_income, "total_debt": total_debt,
                "equity": equity, "tax_rate": tax_rate, "nopat": round(nopat, 2)},
        unit="%",
        interpretation=("Exceptional (>20%)" if val > 20 else
                        "Good (12-20%)" if val > 12 else
                        "Average (8-12%)" if val > 8 else "Below average (<8%)")
    )


def calc_debt_ebitda(total_debt: float, cash: float, ebitda: float) -> CalcResult:
    """Net Debt/EBITDA = (Total Debt - Cash) / EBITDA"""
    if ebitda == 0:
        return CalcResult("Net Debt/EBITDA", 0, "N/A", {})
    net_debt = total_debt - cash
    val = net_debt / ebitda
    return CalcResult(
        name="Net Debt/EBITDA",
        value=round(val, 2),
        formula="(Total Debt - Cash) / EBITDA",
        inputs={"total_debt": total_debt, "cash": cash, "ebitda": ebitda,
                "net_debt": round(net_debt, 2)},
        unit="x",
        interpretation=("Fortress (<1x)" if val < 1 else
                        "Safe (1-2x)" if val < 2 else
                        "Manageable (2-3x)" if val < 3 else
                        "Stretched (3-4x)" if val < 4 else "Dangerous (>4x)")
    )


def calc_var(position_usd: float, vol_annual_pct: float,
             horizon_days: int = 30, confidence: float = 0.95) -> CalcResult:
    """
    VaR = Position × Daily_Vol × Z × sqrt(horizon)
    Daily_Vol = Annual_Vol / sqrt(252)
    """
    z_map  = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z      = z_map.get(confidence, 1.645)
    dv     = (vol_annual_pct / 100) / math.sqrt(252)
    var    = position_usd * dv * z * math.sqrt(horizon_days)
    return CalcResult(
        name=f"VaR ({confidence:.0%}, {horizon_days}d)",
        value=round(var, 2),
        formula=f"Position × (Annual_Vol/√252) × {z} × √{horizon_days}",
        inputs={"position_usd": position_usd, "annual_vol_pct": vol_annual_pct,
                "horizon_days": horizon_days, "confidence": confidence,
                "daily_vol": round(dv, 5), "z_score": z},
        unit="USD",
        interpretation=(f"With {confidence:.0%} confidence, max loss in {horizon_days} days "
                        f"is ${var:,.0f} on a ${position_usd:,.0f} position")
    )


def calc_sharpe(expected_return: float, volatility: float,
                risk_free: float = 0.05) -> CalcResult:
    """Sharpe = (Expected Return - Risk Free Rate) / Volatility"""
    if volatility == 0:
        return CalcResult("Sharpe Ratio", 0, "N/A", {})
    val = (expected_return - risk_free) / volatility
    return CalcResult(
        name="Sharpe Ratio",
        value=round(val, 2),
        formula="(Expected Return - Risk Free Rate) / Volatility",
        inputs={"expected_return": expected_return, "risk_free": risk_free,
                "volatility": volatility},
        unit="",
        interpretation=("Excellent (>2)" if val > 2 else
                        "Good (1-2)" if val > 1 else
                        "Acceptable (0.5-1)" if val > 0.5 else "Poor (<0.5)")
    )


def calc_kelly(win_rate: float, avg_win: float, avg_loss: float) -> CalcResult:
    """
    Kelly % = Win_Rate - (1-Win_Rate) / (Avg_Win/Avg_Loss)
    Use Half-Kelly in practice.
    """
    if avg_loss == 0:
        return CalcResult("Kelly Criterion", 0, "N/A", {})
    full_kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    half_kelly = max(0, full_kelly / 2)
    return CalcResult(
        name="Half-Kelly Position Size",
        value=round(half_kelly * 100, 1),
        formula="[W - (1-W)/(A/L)] / 2, W=win_rate, A=avg_win, L=avg_loss",
        inputs={"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
                "full_kelly": round(full_kelly, 4)},
        unit="% of portfolio",
        interpretation=(f"Optimal position size: {half_kelly*100:.1f}% of portfolio. "
                        "Half-Kelly used for safety margin.")
    )


def calc_dcf(
    base_fcf:     float,
    growth_rates: list[float],   # one per year
    terminal_growth: float,
    wacc:         float,
    shares:       float,
) -> CalcResult:
    """
    DCF = Σ FCF_t/(1+WACC)^t + Terminal_Value/(1+WACC)^n
    Terminal_Value = FCF_n × (1+g) / (WACC - g)
    Intrinsic Value per Share = Total / Shares
    """
    fcf_t  = base_fcf
    pv_sum = 0.0
    for t, g in enumerate(growth_rates, start=1):
        fcf_t  = fcf_t * (1 + g)
        pv_sum += fcf_t / ((1 + wacc) ** t)

    tv  = fcf_t * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / ((1 + wacc) ** len(growth_rates))
    total = pv_sum + pv_tv
    per_share = total / shares if shares > 0 else 0

    return CalcResult(
        name="DCF Intrinsic Value",
        value=round(per_share, 2),
        formula="Σ FCF_t/(1+WACC)^t + [FCF_n×(1+g)/(WACC-g)] / (1+WACC)^n",
        inputs={
            "base_fcf_m":      base_fcf,
            "growth_rates":    growth_rates,
            "terminal_growth": terminal_growth,
            "wacc":            wacc,
            "shares_m":        shares,
            "pv_cashflows_m":  round(pv_sum, 2),
            "terminal_value_m":round(pv_tv, 2),
            "total_equity_m":  round(total, 2),
        },
        unit="USD per share",
        interpretation=(f"Intrinsic value: ${per_share:.2f}/share. "
                        f"TV represents {pv_tv/total*100:.0f}% of total value.")
    )


def calc_margin_of_safety(intrinsic: float, market_price: float) -> CalcResult:
    """Margin of Safety = (Intrinsic Value - Market Price) / Intrinsic Value × 100"""
    if intrinsic == 0:
        return CalcResult("Margin of Safety", 0, "N/A", {})
    val = (intrinsic - market_price) / intrinsic * 100
    return CalcResult(
        name="Margin of Safety",
        value=round(val, 1),
        formula="(Intrinsic Value - Market Price) / Intrinsic Value × 100",
        inputs={"intrinsic": intrinsic, "market_price": market_price},
        unit="%",
        interpretation=("Deep value — Buffett threshold met (>30%)" if val > 30 else
                        "Undervalued (15-30%)" if val > 15 else
                        "Fair value (0-15%)" if val > 0 else
                        f"Overvalued by {abs(val):.1f}%")
    )


def run_full_calc_suite(
    revenue:    float = 0,
    cogs:       float = 0,
    op_income:  float = 0,
    net_income: float = 0,
    total_debt: float = 0,
    cash:       float = 0,
    equity:     float = 0,
    ebitda:     float = 0,
    ticker:     str   = "",
) -> dict:
    """
    Run all calculations at once from extracted financial figures.
    Returns a dict of CalcResult objects ready for the report.
    """
    results = {}
    if revenue and cogs:
        results["gross_margin"]    = calc_gross_margin(revenue, cogs)
    if revenue and op_income:
        results["op_margin"]       = calc_operating_margin(op_income, revenue)
    if revenue and net_income:
        results["net_margin"]      = calc_net_margin(net_income, revenue)
    if op_income and equity:
        results["roic"]            = calc_roic(op_income, total_debt, equity)
    if ebitda and total_debt:
        results["debt_ebitda"]     = calc_debt_ebitda(total_debt, cash, ebitda)
    return results


def calcs_to_dict(calcs: dict) -> dict:
    """Serialize CalcResult objects to JSON-safe dicts."""
    return {
        k: {
            "name":           v.name,
            "value":          v.value,
            "unit":           v.unit,
            "formula":        v.formula,
            "inputs":         v.inputs,
            "interpretation": v.interpretation,
        }
        for k, v in calcs.items()
    }
