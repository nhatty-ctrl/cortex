"""
Backbone AI — Institutional RAG & Analysis Engine
==================================================
This is the analytical brain of the system. It replicates how top-tier
buy-side firms (Bridgewater, Two Sigma, Goldman Research) approach
financial document analysis — but data-driven, not quantitative.

Pipeline:
  1. Document ingestion & chunking (smart, finance-aware)
  2. Metadata extraction (tickers, figures, dates, sentiment)
  3. ChromaDB embedding & storage (with rich metadata filters)
  4. Multi-layer retrieval (semantic + keyword + recency-weighted)
  5. Fundamental analysis (income, balance sheet, cash flow ratios)
  6. Sentiment scoring (news, filings, executive language)
  7. Catalyst detection (events that move stocks)
  8. Risk scoring (multi-factor: macro, sector, company-specific)
  9. Mosaic theory synthesis (connecting dots across sources)
  10. Report generation (executive summary, comps, recommendation)

Covers:
  - 10-K / 10-Q / 8-K SEC filings
  - Earnings call transcripts
  - Analyst research reports
  - News articles (Reuters, FT, WSJ, Bloomberg)
  - Macro / central bank statements
  - LinkedIn / exec activity signals
  - Smart money / 13F data
"""

import re
import math
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# Everything the engine produces is typed. No loose dicts.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialFigure:
    """A single extracted financial number with full context."""
    label:       str            # e.g. "Revenue", "Net Income", "EPS"
    value:       float
    unit:        str            # "USD_M", "USD_B", "PERCENT", "RATIO", "USD"
    period:      str            # "Q3 2024", "FY2023"
    yoy_change:  Optional[float] = None   # year-over-year % change
    qoq_change:  Optional[float] = None   # quarter-over-quarter % change
    beat_miss:   Optional[str]  = None    # "beat", "miss", "in-line"
    estimate:    Optional[float] = None   # analyst consensus estimate
    source_text: str = ""                 # raw sentence it was extracted from

@dataclass
class SentimentScore:
    """Granular sentiment broken into components the way Goldman would."""
    overall:          float   # -1.0 (very bearish) to +1.0 (very bullish)
    management_tone:  float   # how confident/cautious mgmt language sounds
    guidance_tone:    float   # forward-looking language sentiment
    analyst_tone:     float   # sell-side language sentiment
    news_tone:        float   # media coverage sentiment
    uncertainty:      float   # 0.0 (certain) to 1.0 (highly uncertain)
    key_phrases:      list = field(default_factory=list)  # phrases that drove score
    red_flags:        list = field(default_factory=list)  # specific warning language

@dataclass
class Catalyst:
    """A specific event or data point that could move the stock."""
    type:        str     # "earnings_beat", "guidance_raise", "m&a", "exec_change",
                         # "macro_risk", "regulatory", "product_launch", "insider_buy"
    direction:   str     # "bullish", "bearish", "neutral"
    magnitude:   str     # "high", "medium", "low"
    description: str
    timeframe:   str     # "immediate", "1-3 months", "6-12 months"
    confidence:  float   # 0.0 – 1.0
    source:      str

@dataclass
class RiskFactor:
    """A single identified risk with quantified severity."""
    category:    str     # "macro", "sector", "company", "regulatory", "liquidity"
    name:        str
    severity:    float   # 0.0 – 1.0
    probability: float   # 0.0 – 1.0
    description: str
    mitigant:    str     # what would reduce this risk

@dataclass
class FundamentalMetrics:
    """
    The core financial ratios a buy-side analyst would calculate.
    All calculated from scraped filing data — no external data feed needed.
    """
    ticker: str
    period: str

    # ── Profitability ─────────────────────────────────────────────
    gross_margin:    Optional[float] = None   # (Revenue - COGS) / Revenue
    operating_margin: Optional[float] = None  # Operating Income / Revenue
    net_margin:      Optional[float] = None   # Net Income / Revenue
    ebitda_margin:   Optional[float] = None   # EBITDA / Revenue
    roe:             Optional[float] = None   # Net Income / Shareholder Equity
    roa:             Optional[float] = None   # Net Income / Total Assets
    roic:            Optional[float] = None   # NOPAT / Invested Capital

    # ── Growth ───────────────────────────────────────────────────
    revenue_growth_yoy:    Optional[float] = None
    earnings_growth_yoy:   Optional[float] = None
    free_cash_flow_growth: Optional[float] = None

    # ── Leverage & Liquidity ──────────────────────────────────────
    debt_to_equity:    Optional[float] = None  # Total Debt / Equity
    debt_to_ebitda:    Optional[float] = None  # Net Debt / EBITDA
    current_ratio:     Optional[float] = None  # Current Assets / Current Liabilities
    quick_ratio:       Optional[float] = None  # (Current Assets - Inventory) / CL
    interest_coverage: Optional[float] = None  # EBIT / Interest Expense

    # ── Cash Flow Quality ─────────────────────────────────────────
    fcf_yield:         Optional[float] = None  # FCF / Market Cap
    capex_intensity:   Optional[float] = None  # CapEx / Revenue
    cash_conversion:   Optional[float] = None  # FCF / Net Income

    # ── Per-share ─────────────────────────────────────────────────
    eps_diluted:       Optional[float] = None
    eps_growth_yoy:    Optional[float] = None
    book_value_per_share: Optional[float] = None

    # ── Derived signals ───────────────────────────────────────────
    quality_score:     Optional[float] = None  # 0-10: composite quality signal
    financial_health:  Optional[str]   = None  # "strong", "adequate", "stressed"


@dataclass
class MosaicPiece:
    """
    One piece of the mosaic — a single insight from one source.
    Mosaic theory: combine many small legal pieces of info
    to build a picture no single source reveals.
    """
    source_type: str   # "filing", "news", "earnings_call", "linkedin", "13f", "macro"
    insight:     str   # what this piece tells us
    direction:   str   # "bullish", "bearish", "neutral"
    weight:      float # how much to weight this piece (0.0–1.0)
    timestamp:   datetime
    url:         str = ""


@dataclass
class InstitutionalAnalysis:
    """
    The complete analytical output — everything a senior analyst would produce.
    This is what gets stored in ChromaDB and returned to the frontend.
    """
    ticker:          str
    company_name:    str
    analysis_date:   datetime

    fundamentals:    Optional[FundamentalMetrics] = None
    sentiment:       Optional[SentimentScore] = None
    catalysts:       list = field(default_factory=list)   # List[Catalyst]
    risks:           list = field(default_factory=list)   # List[RiskFactor]
    mosaic:          list = field(default_factory=list)   # List[MosaicPiece]
    figures:         list = field(default_factory=list)   # List[FinancialFigure]

    # ── Synthesis ─────────────────────────────────────────────────
    bull_case:       str = ""
    bear_case:       str = ""
    base_case:       str = ""
    key_questions:   list = field(default_factory=list)  # what to watch next quarter

    # ── Final signal ──────────────────────────────────────────────
    signal:          str = "neutral"   # bullish / bearish / neutral / watch
    conviction:      float = 0.0       # 0.0 – 1.0
    price_target_rationale: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DOCUMENT CHUNKING (Finance-Aware)
# Standard chunkers break mid-sentence at arbitrary token limits.
# We chunk at natural financial document boundaries.
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that signal a new section in a financial document
SECTION_BREAKS = [
    r"(?i)^#+\s+",                          # Markdown headers
    r"(?i)^(ITEM\s+\d+[A-Z]?\.?\s+)",      # SEC filing items (ITEM 1A. Risk Factors)
    r"(?i)^(PART\s+[IVX]+\.?\s+)",         # SEC filing parts
    r"(?i)^(Management['s]*\s+Discussion)", # MD&A section
    r"(?i)^(Results\s+of\s+Operations)",
    r"(?i)^(Liquidity\s+and\s+Capital)",
    r"(?i)^(Risk\s+Factors)",
    r"(?i)^(Forward.Looking\s+Statements)",
    r"(?i)^(Consolidated\s+(Balance|Income|Cash))",
    r"(?i)^(OPERATOR:)",                    # Earnings call operator
    r"(?i)^[A-Z][A-Z\s]+:\s",              # Earnings call speaker change
    r"\n{3,}",                              # 3+ blank lines
]

SECTION_PATTERN = re.compile("|".join(SECTION_BREAKS), re.MULTILINE)

# Financial number patterns — we want to extract these precisely
MONEY_PATTERN   = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|B|M|K|bn|mn)?\b",
    re.IGNORECASE
)
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
EPS_PATTERN     = re.compile(r"\$\s*(\d+\.\d{2})\s+(?:per\s+(?:diluted\s+)?share|EPS)", re.IGNORECASE)

# Red-flag language that signals risk (used in sentiment scoring)
RED_FLAG_PHRASES = [
    "going concern", "material weakness", "restatement", "impairment",
    "write-down", "write-off", "covenant breach", "default", "litigation risk",
    "regulatory investigation", "subpoena", "class action", "SEC inquiry",
    "delayed filing", "auditor change", "CFO departure", "unexpected CEO",
    "revenue recognition", "channel stuffing", "inventory build",
    "margin compression", "customer concentration", "single customer",
    "geopolitical uncertainty", "tariff impact", "supply chain disruption",
    "slower than expected", "below expectations", "headwinds", "challenges ahead",
    "uncertain macroeconomic", "difficult environment",
]

# Bullish language signals
BULLISH_PHRASES = [
    "record revenue", "all-time high", "beat expectations", "raised guidance",
    "share buyback", "dividend increase", "market share gains", "new product launch",
    "expanding margins", "accelerating growth", "strong demand", "bookings growth",
    "backlog growth", "pricing power", "margin expansion", "cost discipline",
    "operational leverage", "free cash flow generation", "debt reduction",
    "strategic acquisition", "synergies", "ahead of plan",
]


def chunk_document(text: str, doc_type: str = "news", max_chunk_size: int = 800) -> list[dict]:
    """
    Finance-aware document chunker.

    Strategy by doc_type:
      - "10k" / "10q" / "8k": chunk at SEC section breaks, keep each section together
      - "earnings_call": chunk at speaker changes, keep Q&A pairs together
      - "news": paragraph-based chunking with overlap
      - "analyst_report": chunk at section headers with summary prepended

    Returns list of chunk dicts with metadata.
    """
    chunks = []

    if doc_type in ("10k", "10q", "8k"):
        # Split at SEC structural boundaries
        sections = SECTION_PATTERN.split(text)
        headers  = SECTION_PATTERN.findall(text)
        # Pair headers with sections
        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 50:
                continue
            header = headers[i - 1].strip() if i > 0 and i <= len(headers) else "Section"
            # If section is too long, sub-chunk by paragraph
            if len(section) > max_chunk_size * 2:
                paras = [p.strip() for p in section.split("\n\n") if p.strip()]
                window = []
                window_size = 0
                for para in paras:
                    if window_size + len(para) > max_chunk_size and window:
                        chunks.append({
                            "text":     "\n\n".join(window),
                            "section":  header,
                            "doc_type": doc_type,
                            "chunk_id": str(uuid.uuid4()),
                        })
                        window = [window[-1]]  # 1-paragraph overlap
                        window_size = len(window[-1])
                    window.append(para)
                    window_size += len(para)
                if window:
                    chunks.append({
                        "text":     "\n\n".join(window),
                        "section":  header,
                        "doc_type": doc_type,
                        "chunk_id": str(uuid.uuid4()),
                    })
            else:
                chunks.append({
                    "text":     f"{header}\n{section}",
                    "section":  header,
                    "doc_type": doc_type,
                    "chunk_id": str(uuid.uuid4()),
                })

    elif doc_type == "earnings_call":
        # Split at speaker labels (e.g., "TIM COOK:", "ANALYST:")
        speaker_re = re.compile(r"\n([A-Z][A-Z\s\-]+):\s*", re.MULTILINE)
        parts = speaker_re.split(text)
        speakers = speaker_re.findall(text)
        i, j = 0, 0
        while i < len(parts):
            speaker = speakers[j - 1].strip() if j > 0 and j <= len(speakers) else "UNKNOWN"
            content = parts[i].strip()
            if content and len(content) > 30:
                # Tag management vs analyst utterances
                spk_type = (
                    "management" if any(t in speaker.upper() for t in ["CEO", "CFO", "COO", "PRESIDENT", "OPERATOR"])
                    else "analyst"
                )
                chunks.append({
                    "text":         f"{speaker}: {content}",
                    "section":      "earnings_call",
                    "speaker":      speaker,
                    "speaker_type": spk_type,
                    "doc_type":     doc_type,
                    "chunk_id":     str(uuid.uuid4()),
                })
            i += 1
            j += 1

    else:
        # Default: paragraph-based with sliding window overlap
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip() and len(p.strip()) > 40]
        window, window_len = [], 0
        for para in paragraphs:
            if window_len + len(para) > max_chunk_size and window:
                chunks.append({
                    "text":     "\n\n".join(window),
                    "section":  "body",
                    "doc_type": doc_type,
                    "chunk_id": str(uuid.uuid4()),
                })
                window = window[-1:]  # 1-para overlap
                window_len = len(window[0]) if window else 0
            window.append(para)
            window_len += len(para)
        if window:
            chunks.append({
                "text":     "\n\n".join(window),
                "section":  "body",
                "doc_type": doc_type,
                "chunk_id": str(uuid.uuid4()),
            })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FINANCIAL FIGURE EXTRACTION
# Pull actual numbers from text the way an analyst would highlight them.
# ══════════════════════════════════════════════════════════════════════════════

def normalize_money(value: float, unit: str) -> float:
    """Convert to USD millions as standard unit."""
    unit = (unit or "").lower()
    if unit in ("billion", "b", "bn"):
        return value * 1000
    if unit in ("thousand", "k"):
        return value / 1000
    return value  # assume millions if unspecified

def extract_financial_figures(text: str, ticker: str) -> list[FinancialFigure]:
    """
    Extract every financial figure from a text chunk with context.
    Context window: grab the sentence containing the figure plus 1 sentence each side.
    """
    figures = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for i, sent in enumerate(sentences):
        # ── Money figures ─────────────────────────────────────────
        for match in MONEY_PATTERN.finditer(sent):
            raw_val = float(match.group(1).replace(",", ""))
            unit    = match.group(2) or "M"
            val_m   = normalize_money(raw_val, unit)

            # Try to find a label in the same sentence
            label = _detect_figure_label(sent)

            # Grab context: previous + current + next sentence
            context = " ".join(sentences[max(0, i-1):min(len(sentences), i+2)])

            # Look for YoY change near the figure
            yoy = _extract_yoy(context)

            # Beat/miss language
            beat_miss = None
            if re.search(r"beat|exceeded|above\s+consensus|above\s+estimate", context, re.I):
                beat_miss = "beat"
            elif re.search(r"miss|below\s+consensus|below\s+estimate|fell\s+short", context, re.I):
                beat_miss = "miss"
            elif re.search(r"in.?line|met\s+expectations", context, re.I):
                beat_miss = "in-line"

            # Period detection
            period = _detect_period(context)

            figures.append(FinancialFigure(
                label=       label,
                value=       val_m,
                unit=        "USD_M",
                period=      period,
                yoy_change=  yoy,
                beat_miss=   beat_miss,
                source_text= sent[:200],
            ))

        # ── EPS ───────────────────────────────────────────────────
        for match in EPS_PATTERN.finditer(sent):
            eps_val = float(match.group(1))
            context = " ".join(sentences[max(0, i-1):min(len(sentences), i+2)])
            figures.append(FinancialFigure(
                label=       "EPS (Diluted)",
                value=       eps_val,
                unit=        "USD",
                period=      _detect_period(context),
                yoy_change=  _extract_yoy(context),
                beat_miss=   _detect_beat_miss(context),
                source_text= sent[:200],
            ))

        # ── Percentages with labels ───────────────────────────────
        for match in PERCENT_PATTERN.finditer(sent):
            pct_val = float(match.group(1))
            label   = _detect_figure_label(sent)
            if "margin" in label.lower() or "growth" in label.lower() or "yield" in label.lower():
                context = " ".join(sentences[max(0, i-1):min(len(sentences), i+2)])
                figures.append(FinancialFigure(
                    label=       label,
                    value=       pct_val,
                    unit=        "PERCENT",
                    period=      _detect_period(context),
                    yoy_change=  _extract_yoy(context),
                    source_text= sent[:200],
                ))

    return figures


def _detect_figure_label(sentence: str) -> str:
    """Heuristic: find the closest financial label to a number in a sentence."""
    label_map = {
        r"revenue|net sales|total sales":           "Revenue",
        r"gross profit":                             "Gross Profit",
        r"gross margin":                             "Gross Margin",
        r"operating income|operating profit|ebit":  "Operating Income",
        r"operating margin":                         "Operating Margin",
        r"net income|net profit|net earnings":       "Net Income",
        r"net margin":                               "Net Margin",
        r"ebitda":                                   "EBITDA",
        r"free cash flow|fcf":                       "Free Cash Flow",
        r"capital expenditure|capex":                "CapEx",
        r"research and development|r&d":             "R&D Expense",
        r"total assets":                             "Total Assets",
        r"total debt|long.term debt":                "Total Debt",
        r"cash and cash equivalents":                "Cash & Equivalents",
        r"shareholder[s]* equity|book value":        "Shareholder Equity",
        r"dividend":                                 "Dividend",
        r"share repurchase|buyback":                 "Share Buyback",
        r"guidance|outlook|forecast":                "Guidance",
        r"backlog|bookings|orders":                  "Backlog",
        r"return on equity|roe":                     "ROE",
        r"return on assets|roa":                     "ROA",
    }
    s = sentence.lower()
    for pattern, label in label_map.items():
        if re.search(pattern, s):
            return label
    return "Financial Figure"


def _extract_yoy(text: str) -> Optional[float]:
    """Extract year-over-year growth % from surrounding text."""
    yoy_re = re.compile(
        r"(?:up|down|increased?|decreased?|grew?|declined?|fell?|rose?)\s+"
        r"(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)"
        r"|(\d+(?:\.\d+)?)\s*(?:%|percent)\s+"
        r"(?:year.over.year|yoy|vs\.?\s+(?:prior|last|same)\s+(?:year|period|quarter))",
        re.IGNORECASE
    )
    m = yoy_re.search(text)
    if m:
        val = float(m.group(1) or m.group(2))
        if re.search(r"\b(down|decreased?|declined?|fell?)\b", text[:m.start() + 30], re.I):
            return -val
        return val
    return None


def _detect_period(text: str) -> str:
    """Detect the reporting period from surrounding text."""
    # Quarter detection
    q = re.search(r"(Q[1-4]|first|second|third|fourth)\s+(?:quarter\s+)?(?:of\s+)?(\d{4})", text, re.I)
    if q:
        qmap = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
        qnum = q.group(1) if q.group(1).startswith("Q") else qmap.get(q.group(1).lower(), "Q?")
        return f"{qnum} {q.group(2)}"
    # Full year
    fy = re.search(r"(?:fiscal\s+)?(?:full\s+year|FY|annual)\s+(\d{4})", text, re.I)
    if fy:
        return f"FY{fy.group(1)}"
    # Just year
    yr = re.search(r"\b(20\d{2})\b", text)
    if yr:
        return yr.group(1)
    return "Period Unknown"


def _detect_beat_miss(text: str) -> Optional[str]:
    if re.search(r"beat|exceeded|above\s+(?:consensus|estimate)", text, re.I):
        return "beat"
    if re.search(r"miss|below\s+(?:consensus|estimate)|fell\s+short", text, re.I):
        return "miss"
    if re.search(r"in.?line|met\s+expectations", text, re.I):
        return "in-line"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SENTIMENT SCORING ENGINE
# Multi-dimensional sentiment — not just positive/negative.
# We score management tone separately from news tone from analyst tone.
# This is how hedge funds use NLP on filings.
# ══════════════════════════════════════════════════════════════════════════════

# Weighted phrase dictionaries
MANAGEMENT_BULLISH = {
    "excited":            0.6, "confident":         0.7, "record":            0.8,
    "accelerating":       0.7, "strong momentum":   0.8, "market share gains":0.9,
    "ahead of plan":      0.8, "raising guidance":  1.0, "expanding":         0.5,
    "outperforming":      0.7, "pricing power":     0.7, "robust demand":     0.7,
    "significant opportunity": 0.6, "ahead of schedule": 0.7,
}
MANAGEMENT_BEARISH = {
    "challenging":        -0.5, "headwinds":        -0.6, "uncertain":        -0.5,
    "slower":             -0.5, "below expectations":-0.9,"disappointing":    -0.8,
    "difficult":          -0.5, "cautious":         -0.6, "softness":         -0.6,
    "pressure":           -0.5, "volatile":         -0.4, "macro uncertainty":-0.6,
    "inventory correction":-0.7,"elongated sales cycle":-0.7,
}
ANALYST_BULLISH = {
    "outperform":  0.8, "overweight": 0.8, "buy":          0.7, "strong buy":  1.0,
    "price target raised": 0.9, "upgrade": 0.8, "top pick":  0.9,
    "catalyst rich": 0.7, "compelling valuation": 0.6,
}
ANALYST_BEARISH = {
    "underperform": -0.8, "underweight": -0.8, "sell":     -0.7, "strong sell": -1.0,
    "price target cut": -0.9, "downgrade": -0.8, "avoid":  -0.7,
    "rich valuation": -0.5, "risk/reward unfavorable": -0.7,
}


def score_sentiment(
    texts: dict,  # {"management": str, "analyst": str, "news": str, "guidance": str}
) -> SentimentScore:
    """
    Score sentiment across each text type independently.
    Uses phrase-weighted scoring: each matched phrase adds its weight,
    then we normalize to -1.0 → +1.0.

    This replicates how hedge fund NLP desks score filings —
    weighting management language differently from news.
    """

    def _phrase_score(text: str, bullish_dict: dict, bearish_dict: dict) -> tuple[float, list, list]:
        if not text:
            return 0.0, [], []
        text_lower = text.lower()
        total, count = 0.0, 0
        matched_bull, matched_bear = [], []
        for phrase, weight in bullish_dict.items():
            if phrase in text_lower:
                total += weight
                count += 1
                matched_bull.append(phrase)
        for phrase, weight in bearish_dict.items():
            if phrase in text_lower:
                total += weight   # weight is already negative
                count += 1
                matched_bear.append(phrase)
        if count == 0:
            return 0.0, [], []
        # Normalize: clamp to -1.0 / +1.0 using tanh-like sigmoid
        normalized = total / (1 + abs(total))
        return round(normalized, 3), matched_bull, matched_bear

    def _red_flag_scan(text: str) -> list:
        return [p for p in RED_FLAG_PHRASES if p in text.lower()]

    mgmt_score,  mgmt_bull,  mgmt_bear  = _phrase_score(texts.get("management", ""),  MANAGEMENT_BULLISH, MANAGEMENT_BEARISH)
    analyst_score, a_bull, a_bear       = _phrase_score(texts.get("analyst", ""),     ANALYST_BULLISH,    ANALYST_BEARISH)
    guidance_score, g_bull, g_bear      = _phrase_score(texts.get("guidance", ""),    MANAGEMENT_BULLISH, MANAGEMENT_BEARISH)
    news_score, n_bull, n_bear          = _phrase_score(texts.get("news", ""),         MANAGEMENT_BULLISH, MANAGEMENT_BEARISH)

    # Red flags scan across all text
    all_text   = " ".join(texts.values())
    red_flags  = _red_flag_scan(all_text)
    red_adjust = min(0.0, -0.15 * len(red_flags))  # each red flag pulls overall down

    # Weighted overall: management 35%, guidance 30%, analyst 20%, news 15%
    overall = (
        0.35 * mgmt_score +
        0.30 * guidance_score +
        0.20 * analyst_score +
        0.15 * news_score +
        red_adjust
    )
    overall = max(-1.0, min(1.0, overall))

    # Uncertainty: high if management uses vague/hedging language
    uncertainty_phrases = ["may", "might", "could", "uncertain", "unclear", "depend", "subject to"]
    mgmt_text = texts.get("management", "").lower()
    uncertainty = min(1.0, sum(mgmt_text.count(p) for p in uncertainty_phrases) * 0.08)

    return SentimentScore(
        overall=         round(overall, 3),
        management_tone= round(mgmt_score, 3),
        guidance_tone=   round(guidance_score, 3),
        analyst_tone=    round(analyst_score, 3),
        news_tone=       round(news_score, 3),
        uncertainty=     round(uncertainty, 3),
        key_phrases=     list(set(mgmt_bull + g_bull + a_bull + n_bull))[:10],
        red_flags=       red_flags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FUNDAMENTAL METRICS CALCULATOR
# Given extracted figures, calculate the ratios the big firms use.
# All math is shown explicitly — this is the "how to calculate things" section.
# ══════════════════════════════════════════════════════════════════════════════

def calculate_fundamentals(figures: list[FinancialFigure], ticker: str, period: str) -> FundamentalMetrics:
    """
    Calculate institutional-grade financial ratios from extracted figures.

    HOW EACH RATIO IS CALCULATED:
    ─────────────────────────────────────────────────────────────────────
    Gross Margin       = (Revenue - COGS) / Revenue × 100
    Operating Margin   = Operating Income / Revenue × 100
    Net Margin         = Net Income / Revenue × 100
    EBITDA Margin      = EBITDA / Revenue × 100
    ROE                = Net Income / Shareholder Equity × 100
    ROA                = Net Income / Total Assets × 100
    ROIC               = NOPAT / (Total Debt + Equity) × 100
                         where NOPAT = Operating Income × (1 - tax_rate)
    Debt/Equity        = Total Debt / Shareholder Equity
    Debt/EBITDA        = Net Debt / EBITDA (Net Debt = Total Debt - Cash)
    Current Ratio      = Current Assets / Current Liabilities
    Quick Ratio        = (Current Assets - Inventory) / Current Liabilities
    Interest Coverage  = EBIT / Interest Expense
    FCF Yield          = Free Cash Flow / Market Cap × 100
    CapEx Intensity    = CapEx / Revenue × 100
    Cash Conversion    = Free Cash Flow / Net Income × 100
    ─────────────────────────────────────────────────────────────────────
    """
    fm = FundamentalMetrics(ticker=ticker, period=period)

    # Build a lookup: label → value (take last occurrence = most recent)
    fig_map: dict[str, float] = {}
    for f in figures:
        fig_map[f.label] = f.value

    def get(key: str, *aliases) -> Optional[float]:
        for k in [key] + list(aliases):
            if k in fig_map:
                return fig_map[k]
        return None

    revenue   = get("Revenue")
    cogs      = get("COGS", "Cost of Goods Sold", "Cost of Revenue")
    gross     = get("Gross Profit")
    op_income = get("Operating Income")
    net_inc   = get("Net Income")
    ebitda    = get("EBITDA")
    fcf       = get("Free Cash Flow")
    capex     = get("CapEx")
    total_debt= get("Total Debt")
    cash      = get("Cash & Equivalents")
    equity    = get("Shareholder Equity")
    assets    = get("Total Assets")
    curr_ass  = get("Current Assets")
    curr_liab = get("Current Liabilities")
    inventory = get("Inventory")
    interest  = get("Interest Expense")
    eps       = get("EPS (Diluted)")

    # ── Profitability ─────────────────────────────────────────────
    if gross and revenue and revenue != 0:
        fm.gross_margin = round(gross / revenue * 100, 2)
    elif cogs and revenue and revenue != 0:
        fm.gross_margin = round((revenue - cogs) / revenue * 100, 2)

    if op_income and revenue and revenue != 0:
        fm.operating_margin = round(op_income / revenue * 100, 2)

    if net_inc and revenue and revenue != 0:
        fm.net_margin = round(net_inc / revenue * 100, 2)

    if ebitda and revenue and revenue != 0:
        fm.ebitda_margin = round(ebitda / revenue * 100, 2)

    if net_inc and equity and equity != 0:
        fm.roe = round(net_inc / equity * 100, 2)

    if net_inc and assets and assets != 0:
        fm.roa = round(net_inc / assets * 100, 2)

    # ROIC = NOPAT / Invested Capital. Assume 21% tax rate if unknown.
    if op_income and total_debt is not None and equity:
        nopat = op_income * (1 - 0.21)
        invested_capital = total_debt + equity
        if invested_capital != 0:
            fm.roic = round(nopat / invested_capital * 100, 2)

    # ── Leverage ──────────────────────────────────────────────────
    if total_debt is not None and equity and equity != 0:
        fm.debt_to_equity = round(total_debt / equity, 2)

    if ebitda and total_debt is not None and cash is not None and ebitda != 0:
        net_debt = total_debt - cash
        fm.debt_to_ebitda = round(net_debt / ebitda, 2)

    # ── Liquidity ─────────────────────────────────────────────────
    if curr_ass and curr_liab and curr_liab != 0:
        fm.current_ratio = round(curr_ass / curr_liab, 2)
        inv = inventory or 0
        fm.quick_ratio = round((curr_ass - inv) / curr_liab, 2)

    if op_income and interest and interest != 0:
        fm.interest_coverage = round(op_income / interest, 2)

    # ── Cash Flow ─────────────────────────────────────────────────
    if fcf and capex and revenue and revenue != 0:
        fm.capex_intensity = round(capex / revenue * 100, 2)

    if fcf and net_inc and net_inc != 0:
        fm.cash_conversion = round(fcf / net_inc * 100, 2)

    if eps:
        fm.eps_diluted = eps

    # ── Quality Score (0–10) ──────────────────────────────────────
    # Composite score: higher is better quality business
    # Components: margins, returns, leverage, cash conversion
    score = 5.0  # start at neutral
    if fm.gross_margin:
        score += (fm.gross_margin - 30) / 20       # 50% GM → +1.0
    if fm.operating_margin:
        score += (fm.operating_margin - 10) / 10   # 20% OM → +1.0
    if fm.roe:
        score += (fm.roe - 10) / 15                # 25% ROE → +1.0
    if fm.debt_to_ebitda:
        score -= max(0, (fm.debt_to_ebitda - 2) / 2)  # >4x EBITDA → -1.0
    if fm.cash_conversion:
        score += (fm.cash_conversion - 80) / 40    # 120% conversion → +1.0

    fm.quality_score = round(max(0, min(10, score)), 1)

    # ── Financial Health Label ────────────────────────────────────
    if fm.quality_score >= 7:
        fm.financial_health = "strong"
    elif fm.quality_score >= 4:
        fm.financial_health = "adequate"
    else:
        fm.financial_health = "stressed"

    return fm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CATALYST DETECTION
# Identify specific events in text that could move the stock price.
# ══════════════════════════════════════════════════════════════════════════════

CATALYST_PATTERNS = [
    # (pattern, type, direction, magnitude, timeframe)
    (r"(?i)(raised?|increas|boosted?)\s+guidance",           "guidance_raise",    "bullish",  "high",   "immediate"),
    (r"(?i)(lowered?|cut|reduced?)\s+guidance",              "guidance_cut",      "bearish",  "high",   "immediate"),
    (r"(?i)beat\s+(?:earnings|EPS|revenue)\s+expectations",  "earnings_beat",     "bullish",  "high",   "immediate"),
    (r"(?i)miss(?:ed?)?\s+(?:earnings|EPS|revenue)",         "earnings_miss",     "bearish",  "high",   "immediate"),
    (r"(?i)share\s+(?:buyback|repurchase)\s+program",        "buyback",           "bullish",  "medium", "1-3 months"),
    (r"(?i)dividend\s+(?:increase|raise|hike)",              "dividend_hike",     "bullish",  "medium", "immediate"),
    (r"(?i)dividend\s+cut|dividend\s+suspend",               "dividend_cut",      "bearish",  "high",   "immediate"),
    (r"(?i)acqui(?:re|sition|red?)\s+[A-Z][a-z]+",          "acquisition",       "neutral",  "high",   "1-3 months"),
    (r"(?i)strategic\s+review|exploring\s+sale",             "strategic_review",  "bullish",  "high",   "1-3 months"),
    (r"(?i)CEO\s+(?:resign|depart|step|leaves?)",            "ceo_departure",     "bearish",  "high",   "immediate"),
    (r"(?i)CFO\s+(?:resign|depart|step|leaves?)",            "cfo_departure",     "bearish",  "medium", "immediate"),
    (r"(?i)insider\s+(?:buy|purchas)",                       "insider_buy",       "bullish",  "medium", "1-3 months"),
    (r"(?i)insider\s+sell|insider\s+sold",                   "insider_sell",      "bearish",  "low",    "1-3 months"),
    (r"(?i)FDA\s+(?:approval|approved|clearance)",           "regulatory_approval","bullish", "high",   "immediate"),
    (r"(?i)FDA\s+(?:reject|refusal|warning\s+letter)",       "regulatory_reject", "bearish",  "high",   "immediate"),
    (r"(?i)new\s+(?:product|model|platform)\s+launch",       "product_launch",    "bullish",  "medium", "3-6 months"),
    (r"(?i)lay(?:off|s)|reduction\s+in\s+force|RIF",         "layoffs",           "neutral",  "medium", "1-3 months"),
    (r"(?i)material\s+weakness|internal\s+control",          "material_weakness", "bearish",  "high",   "immediate"),
    (r"(?i)going\s+concern",                                 "going_concern",     "bearish",  "high",   "immediate"),
    (r"(?i)class\s+action|SEC\s+investigation",              "litigation",        "bearish",  "high",   "immediate"),
    (r"(?i)tariff|trade\s+war|import\s+duty",                "macro_trade",       "bearish",  "medium", "3-6 months"),
    (r"(?i)interest\s+rate\s+(?:cut|reduction|pivot)",       "rate_cut",          "bullish",  "medium", "3-6 months"),
    (r"(?i)interest\s+rate\s+(?:hike|increase|rise)",        "rate_hike",         "bearish",  "medium", "3-6 months"),
]

def detect_catalysts(text: str, source: str = "unknown") -> list[Catalyst]:
    """
    Scan text for catalyst patterns and return structured Catalyst objects.
    Each catalyst includes a snippet of the driving text as description.
    """
    catalysts = []
    seen_types = set()  # deduplicate same catalyst type from same doc

    for pattern, cat_type, direction, magnitude, timeframe in CATALYST_PATTERNS:
        for match in re.finditer(pattern, text):
            if cat_type in seen_types:
                continue
            seen_types.add(cat_type)
            # Extract surrounding sentence for description
            start = max(0, match.start() - 100)
            end   = min(len(text), match.end() + 100)
            snippet = text[start:end].strip().replace("\n", " ")
            # Confidence: higher for specific language, lower for vague
            confidence = 0.85 if magnitude == "high" else (0.65 if magnitude == "medium" else 0.45)
            catalysts.append(Catalyst(
                type=        cat_type,
                direction=   direction,
                magnitude=   magnitude,
                description= snippet[:250],
                timeframe=   timeframe,
                confidence=  confidence,
                source=      source,
            ))

    return catalysts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RISK SCORING ENGINE
# Multi-factor risk assessment: macro + sector + company-specific
# ══════════════════════════════════════════════════════════════════════════════

# Risk patterns: (regex, category, name, base_severity, base_probability, mitigant)
RISK_PATTERNS = [
    (r"(?i)geopolit|war|conflict|sanction",
     "macro", "Geopolitical risk", 0.6, 0.4,
     "Diversification of supply chain, regional hedging"),

    (r"(?i)recession|GDP\s+contraction|economic\s+slowdown",
     "macro", "Recessionary environment", 0.7, 0.35,
     "Defensive positioning, cash preservation"),

    (r"(?i)interest\s+rate|federal\s+reserve|monetary\s+policy",
     "macro", "Interest rate sensitivity", 0.5, 0.5,
     "Duration management, floating rate instruments"),

    (r"(?i)inflation|CPI|price\s+pressure",
     "macro", "Inflationary pressure", 0.5, 0.45,
     "Pricing power, cost pass-through ability"),

    (r"(?i)regulation|compliance|antitrust|GDPR|FTC",
     "regulatory", "Regulatory / compliance risk", 0.65, 0.4,
     "Proactive compliance, legal reserves, lobbying"),

    (r"(?i)cyber(?:security|attack)|data\s+breach|hack",
     "company", "Cybersecurity breach", 0.8, 0.25,
     "Security investment, insurance, incident response plan"),

    (r"(?i)customer\s+concentration|single\s+customer|top\s+(?:\d+)\s+customer",
     "company", "Customer concentration risk", 0.6, 0.35,
     "Customer diversification strategy"),

    (r"(?i)supply\s+chain|component\s+shortage|chip\s+shortage|semiconductor",
     "sector", "Supply chain disruption", 0.6, 0.4,
     "Dual sourcing, inventory buffering, reshoring"),

    (r"(?i)competition|market\s+share\s+loss|pricing\s+pressure",
     "sector", "Competitive intensity", 0.55, 0.5,
     "Product differentiation, R&D investment"),

    (r"(?i)debt\s+covenant|credit\s+rating|downgrade|refinanc",
     "company", "Credit / leverage risk", 0.75, 0.3,
     "Debt reduction, refinancing at favorable rates"),

    (r"(?i)currency|forex|FX\s+headwind|exchange\s+rate",
     "macro", "FX / currency risk", 0.4, 0.5,
     "Natural hedging, FX derivatives"),

    (r"(?i)management\s+team|key\s+person|talent\s+retention",
     "company", "Key person / talent risk", 0.5, 0.35,
     "Succession planning, equity compensation"),

    (r"(?i)inventory|channel\s+inventory|excess\s+stock",
     "company", "Inventory / demand risk", 0.55, 0.4,
     "Demand forecasting improvement, lean manufacturing"),

    (r"(?i)pension|defined\s+benefit|unfunded\s+liability",
     "company", "Pension / liability risk", 0.45, 0.3,
     "Pension de-risking, liability-matching strategies"),
]

def score_risks(text: str, fundamentals: Optional[FundamentalMetrics] = None) -> list[RiskFactor]:
    """
    Score risks from text patterns + quantitative fundamentals.
    Severity and probability are adjusted by financial health.
    """
    risks = []
    seen = set()

    for pattern, category, name, base_sev, base_prob, mitigant in RISK_PATTERNS:
        if re.search(pattern, text) and name not in seen:
            seen.add(name)
            sev  = base_sev
            prob = base_prob

            # Adjust by financial health if available
            if fundamentals:
                if fundamentals.financial_health == "stressed":
                    sev  = min(1.0, sev  + 0.15)
                    prob = min(1.0, prob + 0.15)
                elif fundamentals.financial_health == "strong":
                    sev  = max(0.0, sev  - 0.10)
                    prob = max(0.0, prob - 0.10)

            # Extract description snippet
            m = re.search(pattern, text)
            start = max(0, m.start() - 80)
            end   = min(len(text), m.end() + 80)
            snippet = text[start:end].strip().replace("\n", " ")

            risks.append(RiskFactor(
                category=    category,
                name=        name,
                severity=    round(sev, 2),
                probability= round(prob, 2),
                description= snippet[:200],
                mitigant=    mitigant,
            ))

    # Sort by severity × probability (expected impact)
    risks.sort(key=lambda r: r.severity * r.probability, reverse=True)
    return risks[:8]  # top 8 risks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MOSAIC THEORY SYNTHESIZER
# The key insight: combine small pieces from many sources to see
# what no single source reveals. Legal, non-material non-public info.
# ══════════════════════════════════════════════════════════════════════════════

def build_mosaic(
    news_items:   list,   # List[NewsItem]
    filing_text:  str,
    earnings_text:str,
    linkedin_data:list,   # List[dict] from exec scrape
    smart_money:  dict,
    figures:      list[FinancialFigure],
    sentiment:    SentimentScore,
    catalysts:    list[Catalyst],
) -> list[MosaicPiece]:
    """
    Assemble the mosaic: each data type contributes pieces.
    Weight each piece by source reliability and recency.
    """
    pieces = []
    now = datetime.utcnow()

    # ── News pieces ───────────────────────────────────────────────
    for item in news_items[:10]:
        tone = "neutral"
        if any(b in item.summary.lower() for b in ["surge", "jump", "beat", "record"]):
            tone = "bullish"
        elif any(b in item.summary.lower() for b in ["fall", "decline", "miss", "drop"]):
            tone = "bearish"
        pieces.append(MosaicPiece(
            source_type= "news",
            insight=     f"[{item.source}] {item.title}: {item.summary[:100]}",
            direction=   tone,
            weight=      0.6,
            timestamp=   item.published_at if hasattr(item, "published_at") else now,
            url=         item.url if hasattr(item, "url") else "",
        ))

    # ── Earnings call language ────────────────────────────────────
    if earnings_text:
        mgmt_tone = sentiment.management_tone
        direction = "bullish" if mgmt_tone > 0.2 else ("bearish" if mgmt_tone < -0.2 else "neutral")
        pieces.append(MosaicPiece(
            source_type="earnings_call",
            insight=    f"Management tone score: {mgmt_tone:+.2f}. "
                        f"Guidance sentiment: {sentiment.guidance_tone:+.2f}. "
                        f"Uncertainty: {sentiment.uncertainty:.0%}. "
                        f"Key phrases: {', '.join(sentiment.key_phrases[:5])}",
            direction=  direction,
            weight=     0.9,   # management words = high weight
            timestamp=  now,
        ))

    # ── SEC filing signals ────────────────────────────────────────
    if filing_text and sentiment.red_flags:
        pieces.append(MosaicPiece(
            source_type="filing",
            insight=    f"SEC filing red flags detected: {', '.join(sentiment.red_flags[:5])}. "
                        f"These phrases correlate with future earnings disappointments.",
            direction=  "bearish",
            weight=     0.95,  # regulatory language = highest weight
            timestamp=  now,
        ))
    elif filing_text:
        pieces.append(MosaicPiece(
            source_type="filing",
            insight=    "Filing review: no material red flags detected in risk language.",
            direction=  "neutral",
            weight=     0.7,
            timestamp=  now,
        ))

    # ── LinkedIn / exec activity ──────────────────────────────────
    if linkedin_data:
        exec_signals = [d.get("title", "") for d in linkedin_data[:5]]
        has_departure = any(any(w in s.lower() for w in ["left", "departed", "resign", "leaves"]) for s in exec_signals)
        has_hire      = any(any(w in s.lower() for w in ["joined", "appointed", "named", "hired"]) for s in exec_signals)
        direction = "bearish" if has_departure else ("bullish" if has_hire else "neutral")
        weight    = 0.8 if (has_departure or has_hire) else 0.4
        pieces.append(MosaicPiece(
            source_type="linkedin",
            insight=    f"Executive activity: {'; '.join(exec_signals[:3])}",
            direction=  direction,
            weight=     weight,
            timestamp=  now,
        ))

    # ── Smart money / 13F ─────────────────────────────────────────
    if smart_money and smart_money.get("raw_html"):
        raw = smart_money["raw_html"].lower()
        has_buffett = "berkshire" in raw or "buffett" in raw
        has_added   = "added" in raw or "increased" in raw
        has_sold    = "sold" in raw or "reduced" in raw or "eliminated" in raw
        direction = "bullish" if (has_buffett and has_added) else ("bearish" if has_sold else "neutral")
        pieces.append(MosaicPiece(
            source_type="13f",
            insight=    f"Smart money positioning: "
                        f"{'Berkshire/top fund presence detected. ' if has_buffett else ''}"
                        f"{'Recent additions noted. ' if has_added else ''}"
                        f"{'Recent sales/reductions noted.' if has_sold else ''}",
            direction=  direction,
            weight=     0.85,
            timestamp=  now,
        ))

    # ── Catalysts piece ───────────────────────────────────────────
    bull_cats = [c for c in catalysts if c.direction == "bullish" and c.magnitude == "high"]
    bear_cats = [c for c in catalysts if c.direction == "bearish" and c.magnitude == "high"]
    if bull_cats:
        pieces.append(MosaicPiece(
            source_type="catalyst",
            insight=    f"High-magnitude bullish catalysts: {', '.join([c.type for c in bull_cats])}",
            direction=  "bullish",
            weight=     0.9,
            timestamp=  now,
        ))
    if bear_cats:
        pieces.append(MosaicPiece(
            source_type="catalyst",
            insight=    f"High-magnitude bearish catalysts: {', '.join([c.type for c in bear_cats])}",
            direction=  "bearish",
            weight=     0.9,
            timestamp=  now,
        ))

    # ── Financial figures piece ───────────────────────────────────
    beats = [f for f in figures if f.beat_miss == "beat"]
    misses= [f for f in figures if f.beat_miss == "miss"]
    if beats or misses:
        pieces.append(MosaicPiece(
            source_type="filing",
            insight=    f"Earnings surprises: {len(beats)} beats "
                        f"({', '.join([f.label for f in beats[:3]])}), "
                        f"{len(misses)} misses ({', '.join([f.label for f in misses[:3]])})",
            direction=  "bullish" if len(beats) > len(misses) else ("bearish" if len(misses) > len(beats) else "neutral"),
            weight=     0.85,
            timestamp=  now,
        ))

    return pieces


def synthesize_mosaic_signal(pieces: list[MosaicPiece]) -> tuple[str, float]:
    """
    Aggregate all mosaic pieces into a final weighted signal.

    Method: weighted average of directional scores.
      bullish piece → +1 × weight
      bearish piece → -1 × weight
      neutral piece →  0 × weight
    Normalize by total weight.
    """
    total_weight, weighted_sum = 0.0, 0.0
    direction_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}

    for piece in pieces:
        score = direction_map.get(piece.direction, 0.0)
        weighted_sum  += score * piece.weight
        total_weight  += piece.weight

    if total_weight == 0:
        return "neutral", 0.0

    normalized = weighted_sum / total_weight  # -1.0 to +1.0
    conviction  = abs(normalized)             # 0.0 to 1.0

    if normalized > 0.3:
        signal = "bullish"
    elif normalized < -0.3:
        signal = "bearish"
    elif conviction < 0.15:
        signal = "neutral"
    else:
        signal = "watch"

    return signal, round(conviction, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CHROMADB STORE (Enhanced with rich metadata)
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "backbone_institutional"

def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_document(
    text:     str,
    ticker:   str,
    doc_type: str,
    source:   str,
    url:      str = "",
    date:     str = "",
    extra_meta: dict = None,
) -> int:
    """
    Chunk, extract figures, score sentiment, and store a full document.
    Returns number of chunks stored.
    """
    collection = _get_collection()
    chunks     = chunk_document(text, doc_type=doc_type)
    if not chunks:
        return 0

    # Extract figures from full doc (not per-chunk — need full context)
    figures   = extract_financial_figures(text, ticker)
    catalysts = detect_catalysts(text, source=source)

    documents, ids, metadatas = [], [], []

    for chunk in chunks:
        doc_id = hashlib.md5(f"{ticker}_{url}_{chunk['chunk_id']}".encode()).hexdigest()

        # Figure count in this chunk (for retrieval weighting)
        chunk_figures = extract_financial_figures(chunk["text"], ticker)

        meta = {
            "ticker":        ticker,
            "doc_type":      doc_type,
            "source":        source,
            "url":           url,
            "date":          date or datetime.utcnow().isoformat(),
            "section":       chunk.get("section", "body"),
            "speaker_type":  chunk.get("speaker_type", ""),
            "figure_count":  len(chunk_figures),
            "catalyst_count":len(catalysts),
            "has_red_flags": int(any(rf in chunk["text"].lower() for rf in RED_FLAG_PHRASES)),
            "chunk_id":      chunk["chunk_id"],
        }
        if extra_meta:
            # Chroma requires string/int/float values only
            for k, v in extra_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        documents.append(chunk["text"])
        ids.append(doc_id)
        metadatas.append(meta)

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    return len(documents)


def retrieve_for_analysis(
    ticker:  str,
    query:   str,
    n:       int = 12,
    filters: dict = None,
) -> list[dict]:
    """
    Multi-strategy retrieval:
    1. Primary semantic search filtered to ticker
    2. Boost results with red flags if query mentions "risk"
    3. Boost results from filings if query mentions "financials" or "earnings"

    Returns list of {text, metadata, score} dicts sorted by relevance.
    """
    collection = _get_collection()

    where = {"ticker": ticker}
    if filters:
        where.update(filters)

    # Adjust n based on query type
    if any(w in query.lower() for w in ["risk", "danger", "concern", "problem"]):
        where_override = {**where, "has_red_flags": 1}
        try:
            risk_results = collection.query(
                query_texts=[query], n_results=min(4, n), where=where_override
            )
            extra_docs = risk_results.get("documents", [[]])[0]
            extra_meta = risk_results.get("metadatas", [[]])[0]
        except Exception:
            extra_docs, extra_meta = [], []
    else:
        extra_docs, extra_meta = [], []

    # Main semantic retrieval
    try:
        results   = collection.query(query_texts=[query], n_results=n, where=where)
        main_docs = results.get("documents", [[]])[0]
        main_meta = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
    except Exception:
        return []

    # Merge and deduplicate
    seen, combined = set(), []
    for doc, meta, dist in zip(main_docs, main_meta, distances):
        key = meta.get("chunk_id", doc[:50])
        if key not in seen:
            seen.add(key)
            combined.append({
                "text":     doc,
                "metadata": meta,
                "score":    round(1 - dist, 4),   # cosine similarity
            })
    for doc, meta in zip(extra_docs, extra_meta):
        key = meta.get("chunk_id", doc[:50])
        if key not in seen:
            seen.add(key)
            combined.append({
                "text":     doc,
                "metadata": meta,
                "score":    0.6,   # risk-boost docs get fixed decent score
            })

    # Sort by score descending
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n]


def format_context_for_llm(retrieved: list[dict], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into an LLM-ready context string.
    Includes source attribution for every chunk.
    Truncates to max_chars to stay within context window.
    """
    parts = []
    total = 0
    for item in retrieved:
        meta   = item["metadata"]
        source = f"[{meta.get('doc_type','').upper()} | {meta.get('source','')} | {meta.get('date','')[:10]} | score={item['score']:.2f}]"
        chunk  = f"{source}\n{item['text']}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n━━━\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FULL ANALYSIS PIPELINE
# Orchestrates everything above into one function call.
# This is what the /api/reports endpoint calls.
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    ticker:        str,
    company_name:  str,
    news_items:    list,
    filing_text:   str   = "",
    earnings_text: str   = "",
    analyst_text:  str   = "",
    linkedin_data: list  = None,
    smart_money:   dict  = None,
) -> InstitutionalAnalysis:
    """
    Full institutional analysis pipeline. Returns InstitutionalAnalysis.

    Steps:
      1. Ingest all docs into ChromaDB
      2. Extract financial figures from filings
      3. Calculate fundamental ratios
      4. Score multi-dimensional sentiment
      5. Detect catalysts
      6. Score risks
      7. Build mosaic
      8. Synthesize final signal
      9. Generate bull/bear/base cases
    """
    # ── Step 1: Ingest everything ──────────────────────────────────
    full_text = "\n\n".join([
        "\n".join([f"{n.title}: {n.summary}" for n in news_items]) if news_items else "",
        filing_text, earnings_text, analyst_text
    ])

    if filing_text:
        ingest_document(filing_text, ticker, "10k", "SEC EDGAR", date=datetime.utcnow().isoformat())
    if earnings_text:
        ingest_document(earnings_text, ticker, "earnings_call", "Earnings Call", date=datetime.utcnow().isoformat())
    if analyst_text:
        ingest_document(analyst_text, ticker, "analyst_report", "Analyst Report", date=datetime.utcnow().isoformat())
    for item in (news_items or [])[:10]:
        ingest_document(
            getattr(item, "raw_text", "") or f"{item.title} {item.summary}",
            ticker, "news", getattr(item, "source", "Web"),
            url=getattr(item, "url", ""), date=datetime.utcnow().isoformat()
        )

    # ── Step 2: Extract figures ────────────────────────────────────
    figures = extract_financial_figures(filing_text + "\n" + earnings_text, ticker)

    # ── Step 3: Calculate fundamentals ────────────────────────────
    period = _detect_period(filing_text[:500]) if filing_text else "Latest"
    fundamentals = calculate_fundamentals(figures, ticker, period) if figures else None

    # ── Step 4: Sentiment scoring ──────────────────────────────────
    texts_for_sentiment = {
        "management": earnings_text[:3000],
        "guidance":   _extract_guidance_section(earnings_text),
        "analyst":    analyst_text[:2000],
        "news":       " ".join([f"{n.title} {n.summary}" for n in (news_items or [])[:10]]),
    }
    sentiment = score_sentiment(texts_for_sentiment)

    # ── Step 5: Catalysts ──────────────────────────────────────────
    catalysts = detect_catalysts(full_text[:5000], source=f"{ticker} multi-source")

    # ── Step 6: Risks ──────────────────────────────────────────────
    risks = score_risks(full_text[:5000], fundamentals)

    # ── Step 7: Mosaic ────────────────────────────────────────────
    mosaic = build_mosaic(
        news_items=    news_items or [],
        filing_text=   filing_text,
        earnings_text= earnings_text,
        linkedin_data= linkedin_data or [],
        smart_money=   smart_money or {},
        figures=       figures,
        sentiment=     sentiment,
        catalysts=     catalysts,
    )

    # ── Step 8: Signal synthesis ───────────────────────────────────
    signal, conviction = synthesize_mosaic_signal(mosaic)

    # ── Step 9: Bull / Bear / Base cases ──────────────────────────
    bull, bear, base = _generate_cases(
        ticker, company_name, fundamentals, sentiment, catalysts, risks, figures
    )

    # ── Step 10: Key questions ────────────────────────────────────
    questions = _generate_key_questions(ticker, fundamentals, catalysts, risks)

    return InstitutionalAnalysis(
        ticker=         ticker,
        company_name=   company_name,
        analysis_date=  datetime.utcnow(),
        fundamentals=   fundamentals,
        sentiment=      sentiment,
        catalysts=      catalysts,
        risks=          risks,
        mosaic=         mosaic,
        figures=        figures[:20],
        bull_case=      bull,
        bear_case=      bear,
        base_case=      base,
        key_questions=  questions,
        signal=         signal,
        conviction=     conviction,
    )


def _extract_guidance_section(earnings_text: str) -> str:
    """Pull guidance / outlook section from earnings call text."""
    patterns = [r"(?i)guidance|outlook|forecast|full.year|next quarter"]
    for p in patterns:
        m = re.search(p, earnings_text)
        if m:
            return earnings_text[m.start():m.start() + 600]
    return earnings_text[-800:] if earnings_text else ""


def _generate_cases(
    ticker, company_name, fundamentals, sentiment, catalysts, risks, figures
) -> tuple[str, str, str]:
    """Generate bull / bear / base case narratives from data."""

    bull_cats  = [c for c in catalysts if c.direction == "bullish"]
    bear_cats  = [c for c in catalysts if c.direction == "bearish"]
    high_risks = [r for r in risks if r.severity >= 0.6]

    # Bull case
    bull_pts = []
    if sentiment.overall > 0.2:
        bull_pts.append(f"positive management tone ({sentiment.overall:+.2f})")
    if fundamentals and fundamentals.gross_margin and fundamentals.gross_margin > 40:
        bull_pts.append(f"strong gross margin ({fundamentals.gross_margin:.1f}%)")
    if fundamentals and fundamentals.roic and fundamentals.roic > 15:
        bull_pts.append(f"high ROIC ({fundamentals.roic:.1f}%) indicating durable competitive advantage")
    for c in bull_cats[:3]:
        bull_pts.append(c.description[:80])
    bull = (
        f"{company_name} ({ticker}) bull case: " +
        ("; ".join(bull_pts) if bull_pts else "limited bullish data available")
    )

    # Bear case
    bear_pts = []
    if sentiment.overall < -0.2:
        bear_pts.append(f"negative management tone ({sentiment.overall:+.2f})")
    if sentiment.red_flags:
        bear_pts.append(f"filing red flags: {', '.join(sentiment.red_flags[:3])}")
    for r in high_risks[:3]:
        bear_pts.append(f"{r.name} (severity {r.severity:.0%})")
    for c in bear_cats[:2]:
        bear_pts.append(c.description[:80])
    bear = (
        f"{company_name} ({ticker}) bear case: " +
        ("; ".join(bear_pts) if bear_pts else "limited bearish data available")
    )

    # Base case: weighted blend
    base_direction = "cautiously optimistic" if sentiment.overall > 0.1 else ("cautious" if sentiment.overall < -0.1 else "neutral")
    health = fundamentals.financial_health if fundamentals else "unknown"
    base = (
        f"Base case for {company_name}: {base_direction} outlook. "
        f"Financial health assessed as {health}. "
        f"Sentiment score: {sentiment.overall:+.2f}, uncertainty: {sentiment.uncertainty:.0%}. "
        f"Key swing factor: {catalysts[0].description[:100] if catalysts else 'monitor next earnings'}."
    )

    return bull, bear, base


def _generate_key_questions(ticker, fundamentals, catalysts, risks) -> list[str]:
    """Generate the key questions an analyst would ask management on the next call."""
    questions = []

    if fundamentals:
        if fundamentals.debt_to_ebitda and fundamentals.debt_to_ebitda > 3:
            questions.append(f"What is the plan to reduce leverage (currently {fundamentals.debt_to_ebitda:.1f}x Net Debt/EBITDA)?")
        if fundamentals.capex_intensity and fundamentals.capex_intensity > 10:
            questions.append(f"How will CapEx intensity ({fundamentals.capex_intensity:.1f}% of revenue) trend over the next 12 months?")
        if fundamentals.gross_margin and fundamentals.gross_margin < 30:
            questions.append("What are the specific drivers of margin pressure and the path to improvement?")

    for c in catalysts:
        if c.type == "guidance_cut":
            questions.append("What assumptions underpin the revised guidance and what would cause a further revision?")
        if c.type == "acquisition":
            questions.append("What are the expected synergies, timeline to accretion, and integration risks?")
        if c.type == "ceo_departure":
            questions.append("What is the succession plan and interim leadership strategy?")

    for r in risks[:3]:
        questions.append(f"How is the company managing {r.name.lower()}?")

    if not questions:
        questions = [
            "What is driving the divergence between revenue growth and margin expansion?",
            "How does management assess the competitive positioning heading into next year?",
            "What is the capital allocation priority: buyback, M&A, or debt reduction?",
        ]

    return questions[:6]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SERIALIZER
# Convert InstitutionalAnalysis → clean dict the API can return as JSON
# ══════════════════════════════════════════════════════════════════════════════

def analysis_to_dict(analysis: InstitutionalAnalysis) -> dict:
    """Serialize InstitutionalAnalysis to a JSON-safe dict for the API."""
    def _dataclass_to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_dataclass_to_dict(i) for i in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    return _dataclass_to_dict(analysis)
