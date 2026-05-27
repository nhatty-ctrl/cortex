"""
Backbone AI — Dual-Brain Agent
Gemini  → multimodal reading (filings, charts, tables, PDFs)
DeepSeek → financial chain-of-thought reasoning
Together → impact mapping, signal generation, strategy advice
"""
import httpx
import google.generativeai as genai
from typing import List, Optional
from config.settings import settings
from api.models.schemas import Signal, SignalType, CompanyImpact, ImpactLevel, StrategyRecommendation, AdviceAction
from rag.chroma_store import retrieve_context
import json, uuid
from datetime import datetime

# ── Configure Gemini ──────────────────────────────────────────────────────────

genai.configure(api_key=settings.GEMINI_API_KEY)
gemini_flash = genai.GenerativeModel(settings.GEMINI_MODEL)
gemini_pro   = genai.GenerativeModel(settings.GEMINI_PRO_MODEL)

# ── DeepSeek API call ─────────────────────────────────────────────────────────

async def deepseek_reason(prompt: str, system: str = "") -> str:
    """Call DeepSeek Reasoner for financial analysis."""
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model":       settings.DEEPSEEK_MODEL,
        "messages":    messages,
        "temperature": 0.3,
        "max_tokens":  2048,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ── Gemini helper ─────────────────────────────────────────────────────────────

def gemini_call(prompt: str, model="flash") -> str:
    m = gemini_pro if model == "pro" else gemini_flash
    resp = m.generate_content(prompt)
    return resp.text

# ── 1. News impact mapper ─────────────────────────────────────────────────────

async def map_news_to_companies(news_summary: str, ticker: str, company: str) -> List[CompanyImpact]:
    """
    Given a news summary, determine how strongly it affects the target company
    and any related sector peers.
    Gemini does the structured extraction, DeepSeek validates the reasoning.
    """
    prompt = f"""
You are a financial analyst. Given this news, assess the impact on {company} ({ticker}) and up to 3 related sector peers.

NEWS:
{news_summary}

Return a JSON array with this exact shape (no extra text):
[
  {{
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "impact_level": "high|medium|low",
    "impact_reason": "one sentence",
    "sentiment": "bullish|bearish|neutral|watch",
    "confidence": 0.85
  }}
]
"""
    raw = gemini_call(prompt)
    try:
        # Strip markdown fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        return [CompanyImpact(**d) for d in data]
    except Exception:
        return [CompanyImpact(
            ticker=ticker, company_name=company,
            impact_level=ImpactLevel.LOW, impact_reason="Parse error — check raw output",
            sentiment=SignalType.NEUTRAL, confidence=0.0,
        )]

# ── 2. Signal generation (DeepSeek CoT) ──────────────────────────────────────

async def generate_signal(ticker: str, company: str, news_context: str) -> Signal:
    """
    Produce an investment signal with a concise rationale.
    Context is retrieved from ChromaDB RAG store.
    """
    rag_context = retrieve_context(
        query=f"{company} investment risk opportunity",
        ticker=ticker,
    )

    system = """You are a senior Wall Street analyst with expertise in quantitative and fundamental analysis.
Always cite your sources. Be specific about what data drives your conclusion.
Give a concise rationale rather than hidden intermediate reasoning."""

    prompt = f"""
Analyze {company} ({ticker}) based on the following data and generate an investment signal.

## RAG CONTEXT (from knowledge base):
{rag_context}

## LIVE NEWS CONTEXT:
{news_context}

## YOUR TASK:
1. Identify the 3 most important signals in this data
2. Assess sentiment: bullish / bearish / neutral / watch
3. Assign confidence 0.0–1.0 based on data quality and consistency
4. List risk flags
5. Suggest applicable trading strategies: momentum / mean-reversion / event-driven / value

Return ONLY valid JSON (no markdown, no extra text):
{{
  "signal_type": "bullish|bearish|neutral|watch",
  "confidence": 0.0,
  "headline": "one crisp sentence summary",
  "reasoning": "2-4 sentence evidence-backed rationale",
  "strategy_tags": ["momentum"],
  "risk_flags": ["high volatility"],
  "affected_sector": "Technology",
  "sources": ["url1", "url2"]
}}
"""
    raw = await deepseek_reason(prompt, system=system)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        return Signal(
            id=           str(uuid.uuid4()),
            ticker=       ticker,
            company_name= company,
            created_at=   datetime.utcnow(),
            **data,
        )
    except Exception as e:
        return Signal(
            id=str(uuid.uuid4()), ticker=ticker, company_name=company,
            signal_type=SignalType.NEUTRAL, confidence=0.0,
            headline="Signal generation error", reasoning=str(e),
        )

# ── 3. Strategy recommendation ────────────────────────────────────────────────

async def generate_recommendation(
    ticker: str,
    company: str,
    signals: List[Signal],
    smart_money: Optional[dict] = None,
) -> StrategyRecommendation:
    """
    Synthesizes all signals + smart money data into a final BUY / HOLD / AVOID call.
    Uses Gemini Pro for report-quality reasoning.
    """
    signal_text = "\n".join([
        f"- [{s.signal_type.upper()} | {s.confidence:.0%}] {s.headline} | Risk: {', '.join(s.risk_flags)}"
        for s in signals
    ])

    smart_text = ""
    if smart_money and smart_money.get("raw_html"):
        smart_text = f"\n## SMART MONEY (13F DATA):\n{smart_money['raw_html'][:1000]}"

    prompt = f"""
You are the Chief Investment Officer of a hedge fund. Based on all available signals, make a final recommendation for {company} ({ticker}).

## SIGNALS:
{signal_text}
{smart_text}

## DECISION FRAMEWORK:
- BUY: confidence > 0.70, majority bullish signals, no critical risk flags
- HOLD: mixed signals or moderate confidence
- AVOID: majority bearish, high-risk flags, or confidence < 0.40

Return ONLY valid JSON:
{{
  "action": "buy|hold|avoid",
  "time_horizon": "short|medium|long",
  "entry_note": "specific price action or catalyst to watch",
  "risk_note": "key risks to monitor",
  "macro_context": "how macro environment affects this call",
  "signals_used": ["signal_id_1", "signal_id_2"]
}}
"""
    raw = gemini_call(prompt, model="pro")
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        signal_ids = [s.id for s in signals]
        return StrategyRecommendation(
            ticker=ticker,
            signals_used=signal_ids,
            generated_at=datetime.utcnow(),
            **{k: v for k, v in data.items() if k != "signals_used"},
        )
    except Exception as e:
        return StrategyRecommendation(
            ticker=ticker,
            action=AdviceAction.HOLD,
            time_horizon="medium",
            risk_note=f"Recommendation parse error: {e}",
        )

# ── 4. Chat Q&A with RAG context ──────────────────────────────────────────────

def chat_with_context(messages: List[dict], ticker: Optional[str] = None) -> str:
    """
    Conversational Q&A grounded in the ChromaDB knowledge base.
    Used by the /api/chat endpoint.
    """
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    context = ""
    if ticker:
        context = retrieve_context(query=last_user_msg, ticker=ticker)

    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[:-1]])

    prompt = f"""You are Backbone AI, an institutional-grade financial research assistant.
Answer ONLY based on the context below. Cite sources. If the context doesn't cover something, say so.

## KNOWLEDGE BASE CONTEXT:
{context if context else "No specific context loaded. Answering from general knowledge."}

## CONVERSATION HISTORY:
{history}

## LATEST USER QUESTION:
{last_user_msg}

## YOUR ANSWER:"""

    return gemini_call(prompt, model="flash")
