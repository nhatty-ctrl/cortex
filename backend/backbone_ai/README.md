# Cortex AI — Goldman-Grade Financial Intelligence
### Web Data UNLOCKED Hackathon — Track 2: Finance & Market Intelligence

Real-time financial intelligence with 22 specialized AI agents, Bright Data web scraping,
and Goldman Sachs-style institutional reports.

## Quick Start
```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env   # fill in your keys
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reports/generate` | Full Goldman report (all 22 agents) |
| GET  | `/api/signals/{ticker}` | Quick signal for a ticker |
| GET  | `/api/signals/stream` | SSE: live signals → 3D brain |
| GET  | `/api/signals/live-scrape` | SSE: Bright Data activity log |
| GET  | `/api/signals/graph` | Nodes + edges for neuron canvas |
| GET  | `/api/signals/trigger/{ticker}` | Force immediate analysis |
| GET  | `/api/signals/macro/latest` | Global macro news |
| GET  | `/api/signals/latest/{ticker}` | Last report for ticker |
| POST | `/api/scrape/news` | Raw news scrape |
| GET  | `/api/scrape/history?query=` | Browse historical parallels |
| GET  | `/api/scrape/context` | RAG context for citations |
| POST | `/api/chat/` | RAG-grounded Q&A |
| GET/POST/DELETE | `/api/watchlist/` | Manage watchlist |

## Agent Architecture (22 agents, 3 phases)

### Tier 0 — Supervisor (1)
CortexSupervisor — routes, coordinates, phases

### Tier 1 — Data (4, always running)
WebHarvester, ExecWatcher, FilingWatcher, SocialRadar

### Tier 2 — Analysis (8, parallel)
FundamentalAnalyst, CrashPredictor, AlphaCalculator, RiskCalculator,
GeoRiskRadar, SentimentAggregator, SmartMoneyTracker, CurrencyCommodityAnalyst

### Tier 3 — Synthesis (5, sequential)
DCFValuator, HistoricalPlaybook, PeerRelativeValue, HoldWindowEngine, DeepSeekReasoner

### Tier 4 — Output (4)
GoldmanReportWriter, AlertEngine, FactChecker, ChatAgent

## Bright Data Tools Used
- **SERP API** — Reuters, FT, SEC EDGAR, Google News search
- **Web Unlocker** — WhaleWisdom 13F, WSJ, Bloomberg, Al Jazeera
- **Browser API** — X.com, LinkedIn (JS-heavy, requires Playwright)
- **Web Scraper API** — LinkedIn company datasets, SEC structured data
