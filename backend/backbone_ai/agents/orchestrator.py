"""
Backbone AI — 20-Agent Orchestration System
============================================
Matches and exceeds Barebone's 20+ specialized AI analyst agents.
Each agent is a self-contained unit with:
  - A dedicated system prompt (its "personality" and domain expertise)
  - Specific input/output types
  - Gemini Flash for speed OR Gemini Pro for depth
  - DeepSeek Reasoner for any chain-of-thought financial logic

Agent roster (20 agents):
  1.  FundamentalAnalyst      — Buffett-style 4-dimensional analysis
  2.  TechnicalAnalyst        — Multi-timeframe support/resistance, Fibonacci
  3.  SentimentAnalyst        — News + Reddit + analyst consensus aggregator
  4.  EventMapper             — 3-stage headline → affected stocks pipeline
  5.  DCFValuator             — Intrinsic value, peer multiples, margin of safety
  6.  InsiderTracker           — SEC Form 4 + Congressional disclosures
  7.  SmartMoneyTracker        — 13F filings, Buffett/Soros/Ackman positions
  8.  EarningsAnalyst          — Beat/miss analysis, guidance trajectory
  9.  RiskScorer              — Multi-factor risk: macro + sector + company
  10. MacroAnalyst            — Fed, ECB, geopolitical, commodity impacts
  11. CompetitiveIntelligence — Market share, moat analysis, peer comparison
  12. BullBearSynthesizer     — Weighted bull/bear/base case generator
  13. PriceTargetCalculator   — Multi-method target with upside/downside
  14. PortfolioReviewer       — 5-sub-agent portfolio stress test
  15. DividendScreener        — Yield, payout ratio, sustainability scoring
  16. ETFAnalyst              — Holdings overlap, factor exposure, fee analysis
  17. FactChecker             — Cross-source verification of claims
  18. ReportWriter            — Goldman-style institutional report generator
  19. AlertEngine             — Threshold monitoring, anomaly detection
  20. Orchestrator            — Routes queries, coordinates multi-agent runs
"""

import json
import httpx
import asyncio
import google.generativeai as genai
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from config.settings import settings

# ── Configure models ──────────────────────────────────────────────────────────
genai.configure(api_key=settings.GEMINI_API_KEY)
_flash = genai.GenerativeModel(settings.GEMINI_MODEL)
_pro   = genai.GenerativeModel(settings.GEMINI_PRO_MODEL)

# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    agent:      str
    ticker:     str
    output:     dict
    confidence: float          # 0.0 – 1.0
    model_used: str
    duration_ms: int
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error:      Optional[str] = None


class BaseAgent:
    name: str = "base"
    model: Literal["flash", "pro", "deepseek"] = "flash"
    SYSTEM: str = "You are a financial analyst."

    def _gemini(self, prompt: str) -> str:
        m = _pro if self.model == "pro" else _flash
        r = m.generate_content(self.SYSTEM + "\n\n" + prompt)
        return r.text

    async def _deepseek(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system",  "content": self.SYSTEM},
                {"role": "user",    "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers, json=payload,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    def _parse_json(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON safely."""
        clean = raw.strip()
        for fence in ["```json", "```JSON", "```"]:
            clean = clean.replace(fence, "")
        clean = clean.strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw_output": clean, "parse_error": True}

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — FUNDAMENTAL ANALYST
# Buffett-style: growth + quality + financial health + valuation
# 4-dimensional rating system
# ══════════════════════════════════════════════════════════════════════════════

class FundamentalAnalyst(BaseAgent):
    name  = "fundamental_analyst"
    model = "pro"
    SYSTEM = """You are a Warren Buffett-style fundamental analyst. You analyze companies
across four dimensions: (1) Business Quality — competitive moat, pricing power, brand,
switching costs; (2) Growth — revenue trajectory, TAM expansion, margin leverage;
(3) Financial Health — balance sheet strength, cash generation, debt safety;
(4) Valuation — intrinsic value vs market price, margin of safety.

You think in decades, not quarters. You look for durable competitive advantages.
You always ask: "Would I own this business if the stock market closed for 10 years?"
Cite specific data. Never be vague. Rate each dimension 1-10."""

    async def run(self, ticker: str, company: str, context: str, figures: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        figures_text = ""
        if figures:
            figures_text = "\n".join([
                f"  - {f.get('label','?')}: {f.get('value','?')} {f.get('unit','')} "
                f"({f.get('period','')}) {'YoY: '+str(f.get('yoy_change'))+'%' if f.get('yoy_change') else ''}"
                for f in figures[:15]
            ])

        prompt = f"""
Perform a complete Warren Buffett-style fundamental analysis of {company} ({ticker}).

## EXTRACTED FINANCIAL DATA:
{figures_text if figures_text else 'Not available — use context below.'}

## DATA CONTEXT:
{context[:4000]}

## YOUR ANALYSIS — return ONLY valid JSON:
{{
  "moat_analysis": {{
    "moat_type": "wide|narrow|none",
    "moat_sources": ["brand", "switching costs", "network effects", "cost advantage", "intangibles"],
    "moat_durability": "decades|5-10 years|uncertain",
    "moat_score": 8,
    "moat_reasoning": "specific evidence from data"
  }},
  "growth_dimension": {{
    "revenue_growth_trend": "accelerating|stable|decelerating",
    "organic_vs_acquired": "mostly organic|mix|mostly M&A",
    "tam_saturation": "low|medium|high",
    "margin_leverage": "yes|no|mixed",
    "score": 7,
    "key_insight": "one crisp sentence"
  }},
  "quality_dimension": {{
    "roe": null,
    "roic": null,
    "gross_margin": null,
    "fcf_conversion": null,
    "management_quality": "exceptional|good|adequate|poor",
    "capital_allocation": "excellent|good|average|poor",
    "score": 7,
    "key_insight": "one crisp sentence"
  }},
  "health_dimension": {{
    "debt_safety": "fortress|safe|manageable|stretched|dangerous",
    "cash_runway_years": null,
    "interest_coverage": null,
    "liquidity_risk": "very low|low|medium|high",
    "score": 7,
    "key_insight": "one crisp sentence"
  }},
  "valuation_dimension": {{
    "vs_intrinsic_value": "deeply undervalued|undervalued|fair|overvalued|bubble",
    "margin_of_safety": null,
    "earnings_yield": null,
    "peer_discount_premium": "30% discount|10% premium|etc",
    "score": 6,
    "key_insight": "one crisp sentence"
  }},
  "overall_rating": 7.2,
  "buffett_verdict": "Would Buffett buy? Explain in 2-3 sentences citing the biggest factor.",
  "10_year_thesis": "If held for 10 years, what is the most likely outcome?",
  "biggest_risk_to_thesis": "The one thing that could make this thesis wrong."
}}"""

        raw = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = min(1.0, data.get("overall_rating", 5) / 10) if "overall_rating" in data else 0.5
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-pro", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — TECHNICAL ANALYST
# Multi-timeframe support/resistance, Fibonacci, ATR volatility
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalAnalyst(BaseAgent):
    name  = "technical_analyst"
    model = "flash"
    SYSTEM = """You are a proprietary algorithmic technical analyst. You read price action,
volume, and momentum across 5 timeframes (daily, weekly, monthly, quarterly, yearly).
You identify support/resistance using multiple methods: swing highs/lows, Fibonacci retracements
(23.6%, 38.2%, 50%, 61.8%, 78.6%), ATR-based volatility zones, and VWAP.
You do not rely on lagging indicators alone. You read market structure.
Output specific price levels, not vague statements."""

    async def run(self, ticker: str, company: str, context: str, price_data: dict = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        price_info = ""
        if price_data:
            price_info = f"""
Current Price: ${price_data.get('current_price', 'N/A')}
52-week High:  ${price_data.get('high_52w', 'N/A')}
52-week Low:   ${price_data.get('low_52w', 'N/A')}
50-day MA:     ${price_data.get('ma_50', 'N/A')}
200-day MA:    ${price_data.get('ma_200', 'N/A')}
Volume (avg):  {price_data.get('avg_volume', 'N/A')}
"""

        prompt = f"""
Perform multi-timeframe technical analysis for {company} ({ticker}).

{price_info}

## PRICE / NEWS CONTEXT (use for sentiment in price action):
{context[:2000]}

Return ONLY valid JSON:
{{
  "trend": {{
    "primary":   "uptrend|downtrend|sideways",
    "secondary": "uptrend|downtrend|sideways",
    "structure":  "higher highs/lows|lower highs/lows|consolidating"
  }},
  "key_levels": {{
    "strong_support":  [0.0, 0.0],
    "weak_support":    [0.0, 0.0],
    "strong_resistance":[0.0, 0.0],
    "weak_resistance": [0.0, 0.0],
    "fibonacci_levels": {{
      "236": 0.0, "382": 0.0, "500": 0.0, "618": 0.0, "786": 0.0
    }}
  }},
  "volatility": {{
    "atr_estimate":   "high|medium|low",
    "current_regime": "expansion|contraction",
    "vix_context":    "risk-on|risk-off|neutral"
  }},
  "signals": {{
    "momentum":  "bullish|bearish|neutral",
    "volume":    "confirming|diverging|neutral",
    "ma_signal": "above 200MA|below 200MA|at 200MA",
    "rsi_zone":  "overbought|oversold|neutral",
    "macd":      "bullish crossover|bearish crossover|flat"
  }},
  "timeframe_alignment": {{
    "daily":     "bullish|bearish|neutral",
    "weekly":    "bullish|bearish|neutral",
    "monthly":   "bullish|bearish|neutral",
    "overall_alignment": "strong bullish|weak bullish|mixed|weak bearish|strong bearish"
  }},
  "trade_setup": {{
    "type":        "breakout|breakdown|reversal|continuation|none",
    "entry_zone":  [0.0, 0.0],
    "stop_loss":   0.0,
    "target_1":    0.0,
    "target_2":    0.0,
    "risk_reward": "3:1",
    "timeframe":   "short|medium|long"
  }},
  "technical_verdict": "one crisp paragraph summary",
  "confidence":         0.70
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("confidence", 0.6)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — SENTIMENT ANALYST
# Multi-source: Wall Street consensus + Reddit + news + insider pattern
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyst(BaseAgent):
    name  = "sentiment_analyst"
    model = "flash"
    SYSTEM = """You are a multi-source sentiment aggregator for financial markets.
You synthesize Wall Street analyst consensus, social media buzz (Reddit, Twitter/X),
news coverage sentiment, and insider trading patterns into a unified sentiment score.
You weight sources differently: institutional analyst upgrades/downgrades carry more weight
than Reddit posts. Insider buys at 52-week lows are significant. Distinguish between
noise and signal. Output must be specific and cited."""

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        prompt = f"""
Aggregate multi-source sentiment for {company} ({ticker}).

## DATA:
{context[:3500]}

Return ONLY valid JSON:
{{
  "wall_street": {{
    "consensus":      "strong buy|buy|hold|sell|strong sell",
    "buy_pct":        65,
    "hold_pct":       25,
    "sell_pct":       10,
    "avg_pt_upside":  18.5,
    "recent_upgrades": ["Bank of America upgraded to Buy", "..."],
    "recent_downgrades":["Morgan Stanley cut to Hold", "..."],
    "analyst_score":  0.65
  }},
  "news_sentiment": {{
    "overall":       "positive|negative|neutral|mixed",
    "score":         0.45,
    "dominant_theme":"earnings beat|CEO drama|product launch|macro concern|...",
    "top_headlines": ["...", "...", "..."],
    "source_quality":"high|medium|low"
  }},
  "social_sentiment": {{
    "reddit_buzz":   "high|medium|low",
    "reddit_tone":   "bullish|bearish|neutral",
    "unusual_activity":"yes|no",
    "retail_narrative":"what retail is saying about this stock",
    "score":         0.30
  }},
  "insider_pattern": {{
    "recent_buys":   ["{{'name': 'CEO John Doe', 'shares': 10000, 'price': 145.20, 'date': '2025-01'}}"],
    "recent_sells":  [],
    "net_insider_sentiment": "bullish|bearish|neutral",
    "cluster_buying": true,
    "score":         0.70
  }},
  "composite_score": 0.58,
  "composite_label": "moderately bullish",
  "key_contrarian_signal": "What does the smart money say that's different from the crowd?",
  "sentiment_shift": "improving|deteriorating|stable vs 30 days ago"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("composite_score", 0.5)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — EVENT MAPPER
# 3-stage pipeline: headline → affected companies → impact magnitude
# ══════════════════════════════════════════════════════════════════════════════

class EventMapper(BaseAgent):
    name  = "event_mapper"
    model = "flash"
    SYSTEM = """You are an event-driven investment strategist. Given any macro or corporate
news headline, you map the full chain of affected companies — primary (directly affected),
secondary (supply chain, customers, competitors), and tertiary (macro/sector spillover).
For each affected entity, you specify direction (bullish/bearish), magnitude (high/medium/low),
and timeframe (immediate/1-3 months/longer). Think like a Goldman Sachs event-driven desk."""

    async def run(self, ticker: str, company: str, context: str, headlines: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        headline_text = "\n".join([f"- {h}" for h in (headlines or [])[:10]]) if headlines else context[:1500]

        prompt = f"""
Map the investment impact of these events. Primary focus on {company} ({ticker}).

## HEADLINES / EVENTS:
{headline_text}

## ADDITIONAL CONTEXT:
{context[:1500]}

Return ONLY valid JSON:
{{
  "primary_impacts": [
    {{
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "direction": "bullish",
      "magnitude": "high",
      "timeframe": "immediate",
      "mechanism": "why exactly this company is affected",
      "confidence": 0.85
    }}
  ],
  "secondary_impacts": [
    {{
      "ticker": "TSM",
      "company": "TSMC",
      "direction": "bullish",
      "magnitude": "medium",
      "timeframe": "1-3 months",
      "mechanism": "supply chain beneficiary",
      "confidence": 0.60
    }}
  ],
  "sector_impacts": [
    {{
      "sector": "Semiconductors",
      "direction": "bullish",
      "magnitude": "high",
      "reasoning": "..."
    }}
  ],
  "unexpected_impacts": "Counterintuitive effects most investors will miss",
  "event_classification": "macro|earnings|regulatory|geopolitical|m&a|product|management",
  "duration": "one-day event|multi-week theme|multi-month regime change"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.75, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — DCF VALUATOR
# Intrinsic value, peer multiples, margin of safety
# HOW DCF IS CALCULATED:
#   Intrinsic Value = Σ(FCF_t / (1+r)^t) + Terminal Value / (1+r)^n
#   Terminal Value  = FCF_n × (1+g) / (r - g)
#   Margin of Safety = (Intrinsic Value - Market Price) / Intrinsic Value
# ══════════════════════════════════════════════════════════════════════════════

class DCFValuator(BaseAgent):
    name  = "dcf_valuator"
    model = "pro"
    SYSTEM = """You are a valuation expert specializing in DCF modeling and comparable company analysis.
You build multi-scenario DCF models (bull/base/bear case) with explicit assumptions.
You also value companies using peer multiples (EV/EBITDA, P/E, P/FCF, EV/Revenue).
IMPORTANT: Always state your assumptions explicitly. Valuation is only as good as its inputs.
DCF FORMULA:
  Intrinsic Value = Sum of (FCF_year / (1 + WACC)^year) + Terminal Value / (1 + WACC)^years
  Terminal Value  = Final year FCF × (1 + terminal growth) / (WACC - terminal growth)
  Margin of Safety = (Intrinsic Value - Current Price) / Intrinsic Value × 100"""

    async def run(self, ticker: str, company: str, context: str, figures: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        figs = ""
        if figures:
            figs = "\n".join([f"  {f.get('label')}: {f.get('value')} {f.get('unit','')}" for f in figures[:12]])

        prompt = f"""
Build a complete valuation analysis for {company} ({ticker}).

## FINANCIAL FIGURES:
{figs if figs else 'Use context below'}

## CONTEXT:
{context[:3000]}

CALCULATION RULES:
- FCF = Operating Cash Flow - CapEx
- WACC = Typical range 8-12% for large caps, 12-15% for growth stocks
- Terminal growth = 2-3% (GDP growth rate, never exceed WACC)
- EV = Market Cap + Net Debt (or - Net Cash)
- EV/EBITDA: tech 20-30x, consumer 10-15x, industrial 8-12x
- P/FCF: growth tech 30-50x, mature 15-25x

Return ONLY valid JSON:
{{
  "dcf_model": {{
    "assumptions": {{
      "base_fcf_usd_m":     500,
      "growth_year1to5":    "15%",
      "growth_year6to10":   "8%",
      "terminal_growth":    "2.5%",
      "wacc":               "10%",
      "projection_years":   10
    }},
    "scenarios": {{
      "bull": {{
        "intrinsic_value_per_share": 0.0,
        "growth_assumption": "20% for 5 years, 10% next 5",
        "probability": 0.25
      }},
      "base": {{
        "intrinsic_value_per_share": 0.0,
        "growth_assumption": "as above",
        "probability": 0.50
      }},
      "bear": {{
        "intrinsic_value_per_share": 0.0,
        "growth_assumption": "5% for 5 years, 2% next 5",
        "probability": 0.25
      }}
    }},
    "probability_weighted_value": 0.0
  }},
  "peer_multiples": {{
    "ev_ebitda_implied": 0.0,
    "pe_implied":        0.0,
    "pfcf_implied":      0.0,
    "ev_revenue_implied":0.0,
    "sector_avg_ev_ebitda": 0.0,
    "premium_discount_to_peers": "15% premium"
  }},
  "margin_of_safety": {{
    "current_price":   0.0,
    "intrinsic_value": 0.0,
    "mos_percent":     0.0,
    "verdict":         "deep value|undervalued|fair|overvalued|bubble",
    "buffett_threshold":"Buffett buys at >30% MoS — is this above or below?"
  }},
  "price_targets": {{
    "1_year":  0.0,
    "3_year":  0.0,
    "upside_1y": "25%",
    "downside_1y": "-15%"
  }},
  "valuation_confidence": 0.65,
  "key_assumption_risk": "The one assumption that most changes the valuation"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("valuation_confidence", 0.6)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-pro", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — INSIDER & CONGRESS TRACKER
# SEC Form 4 + Congressional Stock Act disclosures
# ══════════════════════════════════════════════════════════════════════════════

class InsiderTracker(BaseAgent):
    name  = "insider_tracker"
    model = "flash"
    SYSTEM = """You are an expert in SEC Form 4 filings and Congressional stock disclosures.
You analyze insider buying and selling patterns to extract actionable signals.
KEY RULES:
  - Insider buys are more informative than sells (executives sell for many reasons: tax, divorce, etc.)
  - Cluster buying (multiple insiders buying in a short window) is the strongest signal
  - Open-market buys are more informative than option exercises
  - Congress trades lag 45 days by law — factor this in
  - Size relative to net worth matters: a $1M buy from a billionaire CEO is weak signal"""

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        prompt = f"""
Analyze insider and Congressional trading activity for {company} ({ticker}).

## DATA:
{context[:3000]}

Return ONLY valid JSON:
{{
  "insider_trades": [
    {{
      "name":       "John Doe",
      "title":      "CEO",
      "type":       "open market buy|open market sell|option exercise|gift",
      "shares":     10000,
      "price":      145.20,
      "value_usd":  1452000,
      "date":       "2025-01-15",
      "signal_strength": "strong|moderate|weak",
      "reason":     "why this trade matters or doesn't"
    }}
  ],
  "congress_trades": [
    {{
      "name":   "Rep. Jane Smith (D-CA)",
      "type":   "purchase|sale",
      "amount": "$15,001-$50,000",
      "date":   "2025-01-10",
      "disclosure_lag_days": 22,
      "note":   "member of [relevant committee if known]"
    }}
  ],
  "cluster_buying": {{
    "detected":     false,
    "window_days":  30,
    "num_insiders": 0,
    "signal":       "no signal|weak buy|strong buy cluster|sell cluster"
  }},
  "form4_summary": {{
    "90d_net_shares_bought": 0,
    "90d_net_value_usd":     0,
    "insider_sentiment":     "bullish|bearish|neutral",
    "trend":                 "accelerating buys|decelerating|stable|accelerating sells"
  }},
  "congress_summary": {{
    "net_direction":         "net buying|net selling|mixed",
    "notable_activity":      "description of most notable trade"
  }},
  "composite_signal":   "strong buy|buy|neutral|sell|strong sell",
  "confidence":         0.70,
  "key_insight":        "The single most important insider/congress signal right now"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("confidence", 0.6)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — SMART MONEY TRACKER
# 13F filings: Buffett, Soros, Ackman, Tepper, Griffin, etc.
# ══════════════════════════════════════════════════════════════════════════════

class SmartMoneyTracker(BaseAgent):
    name  = "smart_money_tracker"
    model = "flash"
    SYSTEM = """You are a 13F filing analyst tracking the world's best investors.
You monitor: Berkshire Hathaway (Buffett), Bridgewater (Dalio), Pershing Square (Ackman),
Appaloosa (Tepper), Tiger Global, Citadel (Griffin), Duquesne (Druckenmiller), Viking.
13F filings are quarterly (45-day lag). New positions are most informative.
Exits (eliminations) signal serious concern. Size changes reveal conviction level.
Cross-reference with public statements and letters for additional context."""

    async def run(self, ticker: str, company: str, context: str, smart_money_html: str = "", **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        data_text = smart_money_html[:2000] if smart_money_html else context[:2000]

        prompt = f"""
Analyze 13F smart money positioning for {company} ({ticker}).

## 13F / INSTITUTIONAL DATA:
{data_text}

Return ONLY valid JSON:
{{
  "top_holders": [
    {{
      "fund":      "Berkshire Hathaway",
      "manager":   "Warren Buffett",
      "shares":    5000000,
      "value_usd_m": 725.0,
      "pct_of_portfolio": 3.2,
      "change_qoq": "new position|increased|decreased|eliminated|unchanged",
      "change_pct": 15.0,
      "conviction": "high|medium|low"
    }}
  ],
  "ownership_changes": {{
    "new_buyers":    ["Fund A", "Fund B"],
    "added_to":      ["Fund C (+20%)"],
    "reduced":       ["Fund D (-15%)"],
    "exited":        ["Fund E — full exit"]
  }},
  "institutional_concentration": {{
    "top10_own_pct": 45.0,
    "total_institutional_pct": 78.0,
    "days_to_cover": 4.2,
    "short_interest_pct": 3.1
  }},
  "smart_money_signal": "strong accumulation|accumulation|neutral|distribution|heavy distribution",
  "confidence": 0.70,
  "best_idea_status": "Is this a 'Best Idea' for any top fund? Evidence?",
  "key_insight": "Most important smart money development right now"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("confidence", 0.65)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 8 — EARNINGS ANALYST
# Beat/miss history, guidance trajectory, estimate revisions
# ══════════════════════════════════════════════════════════════════════════════

class EarningsAnalyst(BaseAgent):
    name  = "earnings_analyst"
    model = "flash"
    SYSTEM = """You are an earnings quality analyst. You look beyond headline EPS.
You analyze: (1) quality of the beat (recurring vs one-time); (2) guidance vs consensus;
(3) estimate revision trends (the direction matters more than the level);
(4) earnings call language shift quarter-over-quarter; (5) operating leverage.
KEY: A company that beats but guides down is usually a sell. A company that misses
but raises guidance is usually a buy. The guidance is the story."""

    async def run(self, ticker: str, company: str, context: str, figures: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        beats = [f for f in (figures or []) if f.get("beat_miss") == "beat"]
        misses= [f for f in (figures or []) if f.get("beat_miss") == "miss"]

        prompt = f"""
Analyze earnings quality and trajectory for {company} ({ticker}).

## EXTRACTED DATA:
Beats:  {[f.get('label') for f in beats[:5]]}
Misses: {[f.get('label') for f in misses[:5]]}

## CONTEXT (earnings call + filings):
{context[:3500]}

Return ONLY valid JSON:
{{
  "latest_quarter": {{
    "eps_actual":    0.0,
    "eps_estimate":  0.0,
    "eps_surprise":  "5.2% beat",
    "revenue_actual_m": 0.0,
    "revenue_estimate_m": 0.0,
    "revenue_surprise": "2.1% beat",
    "quality": "high quality|mixed|low quality (one-time items inflating)",
    "key_items": ["R&D capitalization change", "tax benefit", "etc"]
  }},
  "guidance": {{
    "next_q_revenue_guide": "raised|in-line|lowered|withdrawn",
    "fy_revenue_guide":     "raised|in-line|lowered|withdrawn",
    "next_q_eps_guide":     "raised|in-line|lowered|withdrawn",
    "guidance_vs_consensus":"3% above|in-line|4% below",
    "management_confidence":"high|medium|hedging",
    "guidance_signal":      "bullish|bearish|neutral"
  }},
  "estimate_revisions": {{
    "30d_eps_revision":  "+5% consensus upgrades",
    "30d_rev_revision":  "+2%",
    "trend":             "accelerating upgrades|stable|accelerating downgrades",
    "earnings_momentum": "positive|flat|negative"
  }},
  "call_analysis": {{
    "mgmt_tone_shift":   "more confident|similar|more cautious vs last quarter",
    "key_call_quote":    "Most revealing quote from the call",
    "analyst_concern":   "Biggest push-back from analysts on the call",
    "new_risks_disclosed": ["any new risks mentioned", "..."]
  }},
  "beat_miss_history": {{
    "last_4q_eps":      ["beat", "beat", "miss", "beat"],
    "avg_eps_surprise": "3.2%",
    "consistency":      "consistent beater|inconsistent|consistent misser"
  }},
  "earnings_verdict": "bullish|bearish|neutral",
  "key_metric_to_watch": "The one number that will drive next earnings reaction"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.70, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 9 — RISK SCORER
# Multi-factor: macro + sector + company-specific + liquidity + regulatory
# ══════════════════════════════════════════════════════════════════════════════

class RiskScorer(BaseAgent):
    name  = "risk_scorer"
    model = "flash"
    SYSTEM = """You are a quantitative risk analyst. You score investment risk across
5 categories: macro risk (interest rates, recession, FX), sector risk (competition,
disruption, regulation), company risk (leverage, management, concentration),
liquidity risk (can you exit?), and tail risk (black swan scenarios).
For each risk: assign probability (0-100%) and impact (0-10 scale).
Expected Loss = probability × impact. Prioritize by expected loss, not severity alone."""

    async def run(self, ticker: str, company: str, context: str, fundamentals: dict = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        fund_text = json.dumps(fundamentals, default=str)[:500] if fundamentals else ""

        prompt = f"""
Score all material risks for {company} ({ticker}).

## FUNDAMENTALS:
{fund_text}

## CONTEXT:
{context[:3000]}

Return ONLY valid JSON:
{{
  "risk_register": [
    {{
      "id":          "R1",
      "category":    "macro|sector|company|liquidity|regulatory|tail",
      "name":        "Interest rate sensitivity",
      "probability": 0.45,
      "impact":      7.0,
      "expected_loss": 3.15,
      "description": "specific evidence from context",
      "mitigant":    "what would reduce this risk",
      "timeframe":   "immediate|3-6 months|12+ months"
    }}
  ],
  "composite_risk_score": 6.2,
  "risk_label": "low|moderate|elevated|high|extreme",
  "top_3_risks": ["R1", "R2", "R3"],
  "hidden_risk": "The risk the market is NOT pricing in",
  "stress_scenarios": [
    {{
      "scenario":    "Rate hike 100bps",
      "probability": 0.20,
      "estimated_drawdown": "-25%",
      "recovery_time":      "6-12 months"
    }}
  ],
  "risk_adjusted_verdict": "Given risk level, is the return profile attractive?"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        risk = 1.0 - (data.get("composite_risk_score", 5) / 10)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=risk, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 10 — MACRO ANALYST
# Fed, ECB, geopolitical, commodity price chain effects
# ══════════════════════════════════════════════════════════════════════════════

class MacroAnalyst(BaseAgent):
    name  = "macro_analyst"
    model = "flash"
    SYSTEM = """You are a top-down macro strategist. You connect global macro events
to specific stock impacts. Fed rate decisions, CPI/PCE data, GDP prints, geopolitical
events (wars, sanctions, trade policy), commodity prices (oil, gold, copper as growth indicator),
currency moves — you map all of these to sector and individual stock effects.
Think: Ray Dalio's "machine" framework — how does this one input ripple through the system?"""

    async def run(self, ticker: str, company: str, context: str, macro_news: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        macro_text = "\n".join([f"- {n.get('title','')}: {n.get('summary','')}" for n in (macro_news or [])[:8]])

        prompt = f"""
Assess the macro environment's impact on {company} ({ticker}).

## MACRO NEWS:
{macro_text if macro_text else 'See context below'}

## CONTEXT:
{context[:2500]}

Return ONLY valid JSON:
{{
  "macro_regime": {{
    "cycle_phase":   "early expansion|mid expansion|late cycle|recession|recovery",
    "fed_stance":    "hawkish|neutral|dovish|cutting|hiking",
    "risk_appetite": "risk-on|risk-off|transitioning",
    "usd_trend":     "strengthening|weakening|stable",
    "yield_curve":   "normal|flat|inverted"
  }},
  "macro_impacts_on_ticker": [
    {{
      "factor":    "Rising interest rates",
      "direction": "bearish",
      "magnitude": "medium",
      "channel":   "higher discount rate compresses multiple, floating rate debt costs rise",
      "confidence":0.70
    }}
  ],
  "sector_macro_alignment": {{
    "sector":         "Technology",
    "macro_tailwinds":["AI capex cycle", "rate cuts benefiting growth stocks"],
    "macro_headwinds":["strong USD hurts international revenue", "China risk"],
    "net_alignment":  "tailwind|headwind|neutral"
  }},
  "geopolitical_exposure": {{
    "china_revenue_pct":    15.0,
    "russia_exposure":       "none|minimal|material",
    "supply_chain_risk":    "reshoring needed|partially hedged|minimal",
    "sanctions_risk":       "none|low|medium|high"
  }},
  "commodity_sensitivity": {{
    "oil_sensitivity":    "low|medium|high|direct beneficiary",
    "gold_correlation":   "positive|negative|none",
    "copper_signal":      "economy accelerating|slowing based on copper price"
  }},
  "macro_verdict":  "macro environment is a tailwind|headwind|neutral for this stock",
  "key_macro_watch":"The one macro data point to watch closest for this ticker"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.65, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 11 — COMPETITIVE INTELLIGENCE
# Market share, moat strength, peer comparison
# ══════════════════════════════════════════════════════════════════════════════

class CompetitiveIntelligence(BaseAgent):
    name  = "competitive_intelligence"
    model = "flash"
    SYSTEM = """You are a competitive strategy analyst — Michael Porter meets Wall Street.
You analyze: market share trends (gaining or losing?), competitive moat (Porter's 5 forces),
disruptive threats (could a startup or tech shift destroy this business?), and comparative
advantage vs top 3-5 peers. You also analyze R&D spend vs peers, talent acquisition signals,
and product roadmap differentiation."""

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        prompt = f"""
Analyze competitive positioning of {company} ({ticker}).

## CONTEXT:
{context[:3000]}

Return ONLY valid JSON:
{{
  "market_position": {{
    "market_share_est":      "28%",
    "market_share_trend":    "gaining|stable|losing",
    "market_rank":           1,
    "total_addressable_market_b": 500.0
  }},
  "porters_five_forces": {{
    "supplier_power":   "low|medium|high",
    "buyer_power":      "low|medium|high",
    "competitive_rivalry": "low|medium|high",
    "new_entrants_threat": "low|medium|high",
    "substitutes_threat":  "low|medium|high",
    "overall_attractiveness": "very attractive|attractive|moderate|unattractive"
  }},
  "peer_comparison": [
    {{
      "peer_ticker":  "MSFT",
      "peer_name":    "Microsoft",
      "advantage_vs_peer": "Higher margins, better ecosystem lock-in",
      "disadvantage_vs_peer": "Lower cloud growth rate"
    }}
  ],
  "disruption_risk": {{
    "ai_disruption":   "accelerator|neutral|existential threat",
    "startup_threat":  "low|medium|high — specific companies if known",
    "technology_shift":"at risk from|beneficiary of",
    "timeline":        "2-3 years|5+ years|not near-term"
  }},
  "innovation_pipeline": {{
    "rd_pct_revenue":      8.5,
    "rd_vs_peers":         "above average|average|below average",
    "upcoming_catalysts":  ["product launch Q3", "platform upgrade"],
    "patent_moat":         "strong|moderate|weak"
  }},
  "competitive_verdict": "strengthening moat|stable|weakening moat",
  "sleeper_threat":       "The competitive threat most investors are underestimating"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.65, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 12 — BULL/BEAR SYNTHESIZER (DeepSeek — chain-of-thought)
# Weighted bull/bear/base from all agent outputs
# ══════════════════════════════════════════════════════════════════════════════

class BullBearSynthesizer(BaseAgent):
    name  = "bull_bear_synthesizer"
    model = "deepseek"
    SYSTEM = """You are a senior portfolio manager synthesizing multi-agent research into
investment scenarios. You think probabilistically. You assign explicit probabilities to
bull/bear/base cases. You identify the KEY variable — the single factor that determines
which scenario plays out. You are intellectually honest about uncertainty.
Give a concise rationale. Do not be vague."""

    async def run(self, ticker: str, company: str, context: str, agent_outputs: dict = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        agents_text = ""
        if agent_outputs:
            for agent_name, result in agent_outputs.items():
                if result and hasattr(result, 'output'):
                    agents_text += f"\n### {agent_name.upper()} SIGNAL:\n"
                    agents_text += json.dumps(result.output, default=str)[:400]
                    agents_text += "\n"

        prompt = f"""
Synthesize all research signals into bull/bear/base investment scenarios for {ticker}.

## AGENT SIGNALS:
{agents_text[:3000] if agents_text else context[:3000]}

## YOUR TASK:
Weigh each signal. Assign probabilities. Identify the key variable.

Return ONLY valid JSON:
{{
  "scenarios": {{
    "bull_case": {{
      "probability":   0.30,
      "12m_return":    "+45%",
      "thesis":        "3-4 sentence bull case with specific catalysts",
      "catalyst":      "What needs to happen for bull case to play out",
      "key_assumption":"The most important bull case assumption"
    }},
    "base_case": {{
      "probability":   0.50,
      "12m_return":    "+12%",
      "thesis":        "3-4 sentence base case",
      "catalyst":      "What drives the base case",
      "key_assumption":"Most important base case assumption"
    }},
    "bear_case": {{
      "probability":   0.20,
      "12m_return":    "-25%",
      "thesis":        "3-4 sentence bear case with specific risks",
      "catalyst":      "What triggers the bear case",
      "key_assumption":"Most important bear case assumption"
    }}
  }},
  "key_variable": "The SINGLE factor that determines which scenario plays out",
  "probability_weighted_return": "+14%",
  "asymmetry": "upside scenario has 3x the return of downside loss",
  "conviction": "high|medium|low",
  "final_verdict": "buy|accumulate|hold|reduce|avoid",
  "reasoning_chain": "2-4 sentence rationale for the final verdict"
}}"""

        raw  = await self._deepseek(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = {"high": 0.85, "medium": 0.65, "low": 0.45}.get(data.get("conviction", "medium"), 0.65)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="deepseek-reasoner", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 13 — PRICE TARGET CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class PriceTargetCalculator(BaseAgent):
    name  = "price_target_calculator"
    model = "flash"
    SYSTEM = """You compute price targets using multiple methods and weight them.
Methods: (1) DCF intrinsic value, (2) EV/EBITDA peer multiples, (3) P/E relative valuation,
(4) P/FCF, (5) Analyst consensus. You weight by method reliability for this company type.
Growth stocks: weight FCF and DCF more. Mature companies: weight earnings multiples.
Always provide a range (bear PT / base PT / bull PT) not just a point estimate."""

    async def run(self, ticker: str, company: str, context: str, dcf_output: dict = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        dcf_text = json.dumps(dcf_output, default=str)[:600] if dcf_output else ""

        prompt = f"""
Calculate multi-method price target for {company} ({ticker}).

## DCF OUTPUT:
{dcf_text}

## CONTEXT:
{context[:2000]}

Return ONLY valid JSON:
{{
  "current_price": 0.0,
  "methods": {{
    "dcf":          {{"target": 0.0, "weight": 0.35, "basis": "10-year DCF at 10% WACC"}},
    "ev_ebitda":    {{"target": 0.0, "weight": 0.25, "basis": "18x EV/EBITDA peer median"}},
    "pe_relative":  {{"target": 0.0, "weight": 0.20, "basis": "22x forward P/E vs 25x peer avg"}},
    "pfcf":         {{"target": 0.0, "weight": 0.10, "basis": "25x P/FCF"}},
    "consensus":    {{"target": 0.0, "weight": 0.10, "basis": "Wall Street 12-month consensus"}}
  }},
  "blended_target": 0.0,
  "upside_from_current": "25%",
  "price_range": {{
    "bear": 0.0,
    "base": 0.0,
    "bull": 0.0
  }},
  "time_horizon_months": 12,
  "margin_of_safety": "18%",
  "target_confidence": 0.65
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("target_confidence", 0.6)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 14 — REPORT WRITER (Goldman-style)
# Synthesizes everything into an institutional report
# ══════════════════════════════════════════════════════════════════════════════

class ReportWriter(BaseAgent):
    name  = "report_writer"
    model = "pro"
    SYSTEM = """You are a senior equity research analyst at a top-tier investment bank.
You write institutional-grade research reports in the style of Goldman Sachs, Morgan Stanley,
or Bridgewater. Reports are: (1) precise and data-driven — every claim has evidence;
(2) clearly structured — executive summary → fundamentals → catalysts → valuation →
risks → recommendation; (3) actionable — a portfolio manager reads this and knows exactly
what to do. You never hedge everything. You take a clear view."""

    async def run(self, ticker: str, company: str, context: str, all_outputs: dict = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()

        synthesis = ""
        if all_outputs:
            for name, result in all_outputs.items():
                if result and hasattr(result, 'output'):
                    synthesis += f"\n[{name}]: {json.dumps(result.output, default=str)[:300]}\n"

        prompt = f"""
Write a complete Goldman Sachs-style equity research report for {company} ({ticker}).

## AGENT RESEARCH SYNTHESIS:
{synthesis[:4000] if synthesis else context[:4000]}

Return ONLY valid JSON:
{{
  "report": {{
    "rating":    "Buy|Neutral|Sell",
    "conviction":"High|Medium|Low",
    "price_target": 0.0,
    "current_price": 0.0,
    "upside_downside": "+25%",
    "time_horizon": "12 months",

    "executive_summary": "3 crisp paragraphs. First: what we recommend and why.
Second: key evidence. Third: main risk to thesis.",

    "investment_highlights": [
      "Highlight 1 with specific data point",
      "Highlight 2",
      "Highlight 3"
    ],

    "financial_summary": {{
      "revenue_growth": "narrative",
      "margin_expansion": "narrative",
      "fcf_quality": "narrative",
      "balance_sheet": "narrative"
    }},

    "catalysts": [
      {{"catalyst": "Q3 earnings beat", "timeframe": "2 months", "impact": "bullish"}}
    ],

    "valuation_summary": "DCF + peer comps summary in 2 sentences",

    "key_risks": [
      {{"risk": "name", "probability": "medium", "severity": "high", "mitigant": "..."}}
    ],

    "recommendation_summary": "Final paragraph — clear view, specific action, specific
price levels, specific catalysts to monitor."
  }}
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.80, model_used="gemini-pro", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 15 — DIVIDEND SCREENER
# ══════════════════════════════════════════════════════════════════════════════

class DividendScreener(BaseAgent):
    name  = "dividend_screener"
    model = "flash"
    SYSTEM = """You analyze dividend sustainability and attractiveness. Key metrics:
Payout ratio = Dividends / Net Income (healthy: 30-60%, danger: >80%)
Dividend coverage = FCF / Dividends (healthy: >1.5x)
Dividend growth streak: consecutive years of increases (Dividend Aristocrat = 25+ years)
Yield trap detection: high yield from falling price, not growing dividends."""

    async def run(self, ticker: str, company: str, context: str, figures: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()
        prompt = f"""
Analyze dividend profile for {company} ({ticker}).
Context: {context[:2500]}

Return ONLY valid JSON:
{{
  "dividend_yield":        3.2,
  "annual_dividend":       2.40,
  "payout_ratio_pct":      45.0,
  "fcf_coverage":          2.1,
  "dividend_growth_streak_years": 12,
  "5yr_dividend_cagr":     8.5,
  "sustainability":        "very safe|safe|watch|at risk|likely cut",
  "yield_trap_risk":       false,
  "next_dividend_date":    "2025-03-15",
  "ex_dividend_date":      "2025-03-01",
  "special_dividend_history": "none|occasional|regular",
  "verdict":               "income buy|hold for income|trim|avoid for income",
  "income_investor_score": 7.5,
  "key_risk_to_dividend":  "What could force a cut?"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.70, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 16 — FACT CHECKER
# Cross-source claim verification
# ══════════════════════════════════════════════════════════════════════════════

class FactChecker(BaseAgent):
    name  = "fact_checker"
    model = "flash"
    SYSTEM = """You are an AI fact-checker for financial claims. You cross-reference
claims against multiple sources. You flag: (1) contradictions between sources;
(2) unverified numbers; (3) management spin vs actual data; (4) outdated data presented
as current. You output a trust score for each claim. This is Backbone AI's accuracy layer."""

    async def run(self, ticker: str, company: str, context: str, claims: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()
        claims_text = "\n".join([f"- {c}" for c in (claims or [])[:10]]) if claims else "Extract key claims from context"

        prompt = f"""
Fact-check key claims about {company} ({ticker}).

## CLAIMS TO VERIFY:
{claims_text}

## FULL CONTEXT (multiple sources):
{context[:3000]}

Return ONLY valid JSON:
{{
  "verified_claims": [
    {{
      "claim":         "Revenue grew 15% YoY",
      "verdict":       "verified|partially verified|unverified|contradicted",
      "evidence":      "Source X confirms 14.8% growth; rounding to 15% is accurate",
      "confidence":    0.90,
      "sources_agreed":2,
      "sources_total": 3
    }}
  ],
  "contradictions": [
    {{
      "topic":         "Gross margin",
      "source_a":      "Company says 42%",
      "source_b":      "Bloomberg calculates 39%",
      "likely_cause":  "GAAP vs non-GAAP difference",
      "resolution":    "Use GAAP 39% for comparability"
    }}
  ],
  "unverified_claims": ["claim 1", "claim 2"],
  "overall_accuracy_score": 0.85,
  "management_spin_detected": false,
  "spin_examples": [],
  "data_staleness_flags": ["Revenue figure from Q2, now in Q4 — may be stale"]
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("overall_accuracy_score", 0.75)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 17 — ALERT ENGINE
# Threshold monitoring, anomaly detection
# ══════════════════════════════════════════════════════════════════════════════

class AlertEngine(BaseAgent):
    name  = "alert_engine"
    model = "flash"
    SYSTEM = """You are an anomaly detection system for financial signals. You scan data
for threshold breaches and unusual patterns: volume spikes, sentiment shifts >20pts in 24h,
insider cluster buys, analyst upgrades/downgrades after quiet periods, filing keyword changes
(going concern language appearing for first time), and earnings estimate revisions >5%."""

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()
        prompt = f"""
Scan for alerts and anomalies for {company} ({ticker}).
Context: {context[:3000]}

Return ONLY valid JSON:
{{
  "alerts": [
    {{
      "id":        "A1",
      "type":      "volume_spike|sentiment_shift|insider_cluster|analyst_revision|filing_keyword|guidance_change|price_anomaly",
      "severity":  "critical|high|medium|low",
      "title":     "Short alert title",
      "detail":    "What exactly happened and why it matters",
      "direction": "bullish|bearish|neutral",
      "timestamp": "2025-01-15T10:30:00Z",
      "action":    "What should an investor do about this?"
    }}
  ],
  "watchlist_triggers": {{
    "price_below": 0.0,
    "price_above": 0.0,
    "news_keywords": ["FDA approval", "CEO resign", "earnings beat"]
  }},
  "alert_summary": "X critical, Y high, Z medium alerts detected"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        n_alerts = len(data.get("alerts", []))
        severity_map = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        conf = max([severity_map.get(a.get("severity","low"), 0.4) for a in data.get("alerts", [{"severity":"low"}])])
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 18 — PORTFOLIO REVIEWER
# 5 sub-lenses: growth + risk + income + sector + momentum
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioReviewer(BaseAgent):
    name  = "portfolio_reviewer"
    model = "pro"
    SYSTEM = """You review portfolios through 5 simultaneous lenses:
1. GROWTH ANALYST:     Does this holding accelerate portfolio growth?
2. RISK MANAGER:       Does it add or reduce portfolio risk? Correlation to existing holdings?
3. INCOME SPECIALIST:  Dividend and income contribution. Sustainable?
4. SECTOR EXPERT:      Sector concentration and diversification quality
5. MOMENTUM ANALYST:   Is price momentum aligned with fundamentals?
You output each lens separately then give a combined portfolio score."""

    async def run(self, ticker: str, company: str, context: str, portfolio: list = None, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()
        portfolio_text = json.dumps(portfolio, default=str)[:800] if portfolio else "Single stock analysis"

        prompt = f"""
Review {company} ({ticker}) as a portfolio holding.

## EXISTING PORTFOLIO:
{portfolio_text}

## STOCK CONTEXT:
{context[:2500]}

Return ONLY valid JSON:
{{
  "growth_lens":    {{ "score": 7.5, "verdict": "bullish|neutral|bearish", "insight": "..." }},
  "risk_lens":      {{ "score": 6.0, "verdict": "reduces|neutral|adds risk",
                       "beta_estimate": 1.2, "correlation_to_sp500": 0.75, "insight": "..." }},
  "income_lens":    {{ "score": 4.0, "verdict": "income positive|neutral|negative",
                       "yield": 1.2, "income_growth": "5% annually", "insight": "..." }},
  "sector_lens":    {{ "score": 7.0, "sector": "Technology",
                       "concentration_warning": false, "insight": "..." }},
  "momentum_lens":  {{ "score": 8.0, "price_momentum": "bullish|neutral|bearish",
                       "fundamental_momentum": "improving|stable|deteriorating", "insight": "..." }},
  "portfolio_fit_score": 7.0,
  "position_sizing": "core (5-10%)|satellite (2-5%)|speculative (<2%)|avoid",
  "combined_verdict": "add|hold|trim|exit",
  "portfolio_impact": "Adding this improves/hurts portfolio because..."
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        conf = data.get("portfolio_fit_score", 5) / 10
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=conf, model_used="gemini-pro", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 19 — ETF ANALYST
# ══════════════════════════════════════════════════════════════════════════════

class ETFAnalyst(BaseAgent):
    name  = "etf_analyst"
    model = "flash"
    SYSTEM = """You analyze ETFs for factor exposure, holdings overlap, fee efficiency,
and fit vs individual stock alternatives. Key: expense ratio must be justified by alpha
or diversification. Hidden concentrations (Top 10 holdings >60%) defeat diversification.
Factor exposure: value/growth/quality/momentum/size tilt. Liquidity: AUM and bid-ask spread."""

    async def run(self, ticker: str, company: str, context: str, **kwargs) -> AgentResult:
        t0 = datetime.utcnow()
        prompt = f"""
Analyze ETF {ticker} ({company}).
Context: {context[:2500]}

Return ONLY valid JSON:
{{
  "basics": {{
    "aum_b":          50.0,
    "expense_ratio":  0.03,
    "avg_daily_volume_m": 500.0,
    "bid_ask_spread": "0.01%",
    "inception_date": "2010-01-01"
  }},
  "holdings": {{
    "num_holdings":    500,
    "top10_pct":       28.0,
    "top_holdings":    ["AAPL 7%", "MSFT 6.5%", "..."],
    "concentration_risk": "low|medium|high"
  }},
  "factor_exposure": {{
    "value":    0.2,
    "growth":   0.6,
    "quality":  0.5,
    "momentum": 0.4,
    "size":     "large cap blend"
  }},
  "performance": {{
    "1yr_return":  12.5,
    "3yr_return":  35.0,
    "5yr_return":  85.0,
    "vs_benchmark":"1.2% above|below S&P 500"
  }},
  "verdict": "buy|hold|avoid",
  "best_for": "passive long-term investor|tactical allocation|income|growth",
  "alternative": "Consider SPY if you want broader market; QQQ for tech focus"
}}"""

        raw  = self._gemini(prompt)
        data = self._parse_json(raw)
        ms   = int((datetime.utcnow() - t0).total_seconds() * 1000)
        return AgentResult(agent=self.name, ticker=ticker, output=data,
                           confidence=0.65, model_used="gemini-flash", duration_ms=ms)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 20 — ORCHESTRATOR
# Routes queries, runs agents in parallel, returns unified output
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    The master coordinator. Decides which agents to run for a given query type,
    runs them in parallel (where possible), and merges outputs.

    Query routing:
      "full_report"    → all agents
      "quick_signal"   → fundamental + technical + sentiment + bull_bear
      "event_impact"   → event_mapper + macro + risk
      "valuation"      → dcf + price_target + fundamental (quality dims)
      "insider_check"  → insider + smart_money
      "portfolio_add"  → portfolio_reviewer + risk + dividend
    """

    QUICK_AGENTS  = ["fundamental_analyst", "technical_analyst", "sentiment_analyst", "bull_bear_synthesizer"]
    FULL_AGENTS   = [
        "fundamental_analyst", "technical_analyst", "sentiment_analyst",
        "event_mapper", "dcf_valuator", "insider_tracker", "smart_money_tracker",
        "earnings_analyst", "risk_scorer", "macro_analyst", "competitive_intelligence",
        "bull_bear_synthesizer", "price_target_calculator", "fact_checker",
        "alert_engine", "report_writer",
    ]
    EVENT_AGENTS  = ["event_mapper", "macro_analyst", "risk_scorer", "sentiment_analyst"]
    VALUE_AGENTS  = ["dcf_valuator", "price_target_calculator", "fundamental_analyst"]
    INSIDER_AGENTS= ["insider_tracker", "smart_money_tracker"]
    PORTFOLIO_AGENTS = ["portfolio_reviewer", "risk_scorer", "dividend_screener"]

    AGENT_MAP = {
        "fundamental_analyst":      FundamentalAnalyst,
        "technical_analyst":        TechnicalAnalyst,
        "sentiment_analyst":        SentimentAnalyst,
        "event_mapper":             EventMapper,
        "dcf_valuator":             DCFValuator,
        "insider_tracker":          InsiderTracker,
        "smart_money_tracker":      SmartMoneyTracker,
        "earnings_analyst":         EarningsAnalyst,
        "risk_scorer":              RiskScorer,
        "macro_analyst":            MacroAnalyst,
        "competitive_intelligence": CompetitiveIntelligence,
        "bull_bear_synthesizer":    BullBearSynthesizer,
        "price_target_calculator":  PriceTargetCalculator,
        "report_writer":            ReportWriter,
        "dividend_screener":        DividendScreener,
        "fact_checker":             FactChecker,
        "alert_engine":             AlertEngine,
        "portfolio_reviewer":       PortfolioReviewer,
        "etf_analyst":              ETFAnalyst,
    }

    def _get_agent_list(self, mode: str) -> list[str]:
        return {
            "full_report":   self.FULL_AGENTS,
            "quick_signal":  self.QUICK_AGENTS,
            "event_impact":  self.EVENT_AGENTS,
            "valuation":     self.VALUE_AGENTS,
            "insider":       self.INSIDER_AGENTS,
            "portfolio":     self.PORTFOLIO_AGENTS,
        }.get(mode, self.QUICK_AGENTS)

    async def run(
        self,
        ticker:     str,
        company:    str,
        context:    str,
        mode:       str = "quick_signal",
        extra_data: dict = None,
    ) -> dict:
        """
        Run the appropriate agent squad in parallel.
        Returns dict of {agent_name: AgentResult}.
        """
        extra_data = extra_data or {}
        agent_names = self._get_agent_list(mode)

        # Phase 1: Run independent agents in parallel
        # (bull_bear_synthesizer and report_writer need other outputs — run last)
        phase1 = [n for n in agent_names if n not in ("bull_bear_synthesizer", "report_writer")]
        phase2 = [n for n in agent_names if n in ("bull_bear_synthesizer", "report_writer")]

        async def run_one(name: str, prior_outputs: dict = None) -> tuple[str, AgentResult]:
            cls     = self.AGENT_MAP.get(name)
            if not cls:
                return name, None
            agent   = cls()
            kwargs  = {**extra_data}
            if prior_outputs:
                kwargs["agent_outputs"] = prior_outputs
                kwargs["all_outputs"]   = prior_outputs
            try:
                result = await agent.run(ticker=ticker, company=company, context=context, **kwargs)
            except Exception as e:
                result = AgentResult(agent=name, ticker=ticker, output={"error": str(e)},
                                     confidence=0.0, model_used="error", duration_ms=0, error=str(e))
            return name, result

        # Phase 1 in parallel
        phase1_tasks   = [run_one(n) for n in phase1]
        phase1_results = await asyncio.gather(*phase1_tasks)
        phase1_map     = {name: result for name, result in phase1_results}

        # Phase 2 sequentially (needs phase 1 outputs)
        phase2_map: dict = {}
        for name in phase2:
            _, result = await run_one(name, prior_outputs=phase1_map)
            phase2_map[name] = result

        all_results = {**phase1_map, **phase2_map}

        # Build unified summary
        summary = self._summarize(ticker, company, all_results, mode)
        return {"results": {k: asdict(v) if v else None for k, v in all_results.items()},
                "summary": summary,
                "mode":    mode,
                "ticker":  ticker,
                "company": company,
                "timestamp": datetime.utcnow().isoformat()}

    def _summarize(self, ticker: str, company: str, results: dict, mode: str) -> dict:
        """Create a top-level summary card from all agent outputs."""
        verdicts  = []
        risks     = []
        conf_vals = []

        for name, result in results.items():
            if result and not result.error:
                conf_vals.append(result.confidence)
                out = result.output

                # Extract top-level signals
                for key in ("signal_type","final_verdict","verdict","earnings_verdict",
                            "competitive_verdict","technical_verdict","macro_verdict"):
                    if key in out:
                        verdicts.append(str(out[key]))

                # Extract risk labels
                if "risk_label" in out:
                    risks.append(out["risk_label"])

        bull_count  = sum(1 for v in verdicts if any(b in v.lower() for b in ["bull","buy","strong buy","positive"]))
        bear_count  = sum(1 for v in verdicts if any(b in v.lower() for b in ["bear","sell","avoid","negative"]))
        avg_conf    = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else 0.5
        overall     = "bullish" if bull_count > bear_count else ("bearish" if bear_count > bull_count else "neutral")

        report_out  = results.get("report_writer")
        bull_bear   = results.get("bull_bear_synthesizer")

        return {
            "overall_signal":     overall,
            "average_confidence": avg_conf,
            "bull_signals":       bull_count,
            "bear_signals":       bear_count,
            "agents_run":         len(results),
            "final_verdict":      bull_bear.output.get("final_verdict") if bull_bear and not bull_bear.error else overall,
            "price_target":       (report_out.output.get("report", {}).get("price_target")
                                  if report_out and not report_out.error else None),
            "rating":             (report_out.output.get("report", {}).get("rating")
                                  if report_out and not report_out.error else None),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION — call from the API router
# ══════════════════════════════════════════════════════════════════════════════

async def run_agents(
    ticker:  str,
    company: str,
    context: str,
    mode:    str = "quick_signal",
    extra:   dict = None,
) -> dict:
    """Entry point for the /api/signals and /api/reports endpoints."""
    orchestrator = Orchestrator()
    return await orchestrator.run(ticker, company, context, mode=mode, extra_data=extra or {})
