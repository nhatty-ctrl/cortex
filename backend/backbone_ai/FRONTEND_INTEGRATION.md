# Backbone AI — Frontend Integration Guide
# Base URL: http://localhost:8000  (or your deployed URL)

## ── Endpoints your Google AI Studio UI should call ──────────────────────────

### 1. Generate full report (main CTA button)
POST /api/reports/generate
Body:  { "ticker": "AAPL", "company": "Apple Inc.", "depth": "deep" }
Returns: Report object with:
  - executive_summary  → show in main report panel
  - signals[]          → signal cards (bullish/bearish + confidence bar)
  - recommendation     → BUY/HOLD/AVOID card with reasoning
  - news_digest[]      → news feed with source badges
  - smart_money        → company impact table

### 2. Quick live signal (watchlist refresh)
GET /api/signals/{ticker}?company=Apple+Inc.
Returns: Signal object → use for the live feed ticker cards

### 3. Macro / world news panel
GET /api/signals/macro/latest
Returns: List[NewsItem] → global context panel

### 4. Chat / Q&A
POST /api/chat/
Body:  { "messages": [...], "ticker": "AAPL", "use_rag": true }
Returns: { "role": "assistant", "content": "..." }
→ Multi-turn: send full history each call

### 5. Raw news feed
POST /api/scrape/news
Body:  { "ticker": "AAPL", "company": "Apple Inc.", "depth": "quick" }
Returns: List[NewsItem]

### 6. Watchlist CRUD
GET    /api/watchlist/
POST   /api/watchlist/        body: { "ticker": "AAPL", "company_name": "Apple" }
DELETE /api/watchlist/{ticker}

## ── Signal object shape ──────────────────────────────────────────────────────
{
  "id": "uuid",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "signal_type": "bullish",          # bullish | bearish | neutral | watch
  "confidence": 0.82,                # 0.0 – 1.0  → render as % bar
  "headline": "Strong earnings beat driven by services growth",
  "reasoning": "Full DeepSeek CoT text...",
  "strategy_tags": ["momentum", "event-driven"],
  "risk_flags": ["macro headwinds", "China exposure"],
  "sources": ["https://...", "https://..."],
  "affected_sector": "Technology",
  "created_at": "2025-05-25T10:00:00Z"
}

## ── Recommendation object shape ─────────────────────────────────────────────
{
  "ticker": "AAPL",
  "action": "buy",                   # buy | hold | avoid
  "time_horizon": "medium",          # short | medium | long
  "entry_note": "Watch for $180 support confirmation",
  "risk_note": "China revenue risk and macro slowdown",
  "macro_context": "Fed pivot likely to benefit tech in H2",
  "generated_at": "2025-05-25T10:00:00Z"
}

## ── CORS ─────────────────────────────────────────────────────────────────────
# Already configured to allow all origins.
# If AI Studio is at https://your-app.web.app, set:
# CORS allow_origins=["https://your-app.web.app"] in main.py

## ── Run locally ──────────────────────────────────────────────────────────────
# pip install -r requirements.txt
# cp .env.example .env   # fill in your keys
# uvicorn main:app --reload --port 8000

## ── API docs (auto-generated) ────────────────────────────────────────────────
# http://localhost:8000/docs   ← Swagger UI — show this during demo!
