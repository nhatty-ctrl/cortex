# CORTEX // Market Intelligence Engine

Cortex is an autonomous, event-driven financial intelligence engine powered by a 22-agent swarm. It intercepts market alpha via zero-cache web scraping infrastructure (Bright Data) and delivers real-time, mathematical investment intelligence up to six hours before legacy data networks update.

Cortex is designed to capture critical structural shifts—such as unannounced executive transitions, dark pool anomalies, and regulatory front-running markers—before they register on legacy institutional data terminals.

The system orchestrates an asynchronous, multi-layered swarm of 22 specialized autonomous agents to ingest, validate, score, and synthesize unstructured web data into actionable financial notes with zero black boxes.

## ⚡ Core Architecture

```
[ Unstructured Web Data Sources ]
               │
               ▼  (SERP API / Web Unlocker / Scraping Browser)
┌────────────────────────────────────────────────────────┐
│               BRIGHT DATA INGESTION CORE               │
└────────────────────────────────────────────────────────┘
               │
               ▼  (Parallel JSON Payloads via FastAPI)
┌────────────────────────────────────────────────────────┐
│              22-AGENT ASYNCHRONOUS SWARM              │
│  - FilingWatcher         - SocialRadar                 │
│  - ExecWatcher           - MacroTracker                │
└────────────────────────────────────────────────────────┘
               │
               ▼  (Context Routing via ChromaDB)
┌────────────────────────────────────────────────────────┐
│            DEEPSEEK REASONER SYNTHESIS LAYER           │
│    Outlines exact logical verification tracing CoT    │
└────────────────────────────────────────────────────────┘
               │
               ▼  (Real-Time Server-Sent Events)
┌────────────────────────────────────────────────────────┐
│     GOLDMAN-GRADE REPORT & LIVE COMMAND DASHBOARD      │
└────────────────────────────────────────────────────────┘
```

## ✨ Key Features

**Zero-Cache Ingestion**: Bypasses content delivery network (CDN) caches using the Bright Data Web Unlocker and SERP API to scan SEC EDGAR filings, macro indexes, and financial media continuously.

**Structural Delta Isolation**: Tracks changes in corporate directories via JavaScript rendering in the Bright Data Scraping Browser, capturing core leadership changes hours before traditional 8-K filings hit news feeds.

**22-Agent Multi-Layer Swarm**: Executes parallel asset evaluations split into data harvesting nodes, fundamental analysis clusters, and risk assessment engines.

**Transparent Quantitative Verification**: Eliminates LLM "black box" behavior. Every generated report outputs verified mathematical proofs for asset evaluations:

- Return on Invested Capital (ROIC): $\text{ROIC} = \frac{\text{NOPAT}}{\text{Invested Capital}}$

- Intrinsic Discounted Cash Flow (DCF): $\text{Value} = \sum_{t=1}^{n} \frac{\text{FCF}_t}{(1 + \text{WACC})^t} + \frac{\text{Terminal Value}}{(1 + \text{WACC})^n}$

- Value at Risk (VaR): $\text{VaR}_{95\%} = \mu - 1.645 \cdot \sigma$

**Event-Driven UI Streaming**: Employs Server-Sent Events (SSE) to stream a continuous platform "Consciousness Stream" log directly into a dark, high-density dashboard grid without polling lag.

## 🛠️ Technology Stack

**Infrastructure / Data Scraping**: Bright Data (SERP API, Web Unlocker, Scraping Browser, Web Scraper API)

**Backend Runtime**: Python 3.11+, FastAPI (Asynchronous execution framework)

**Vector Engine**: ChromaDB (Low-latency localized similarity context search)

**Agent Intelligence**: DeepSeek Reasoner (Strict Chain-of-Thought logical auditing), Gemini 2.5 Pro (Comprehensive macro report synthesis), Gemini 2.0 Flash (Ultra-high-speed ingest parsing)

**Frontend UI Layout**: HTML5, Tailwind CSS, Native JS, Server-Sent Events

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ (for frontend)
- Bright Data API Credentials
- DeepSeek & Google AI Studio API Keys

### Installation & Local Setup

Clone the repository:
```bash
git clone https://github.com/nhatty-ctrl/cortex.git
cd cortex
```

Configure environment variables:
Create a `.env` file in the `backend/` directory and append your access keys:
```
BRIGHT_DATA_API_KEY="your_bright_data_key_here"
BRIGHT_DATA_ZONE_PROXY="your_proxy_zone_endpoint"
DEEPSEEK_API_KEY="your_deepseek_key_here"
GEMINI_API_KEY="your_gemini_key_here"
DATABASE_URL="sqlite:///./cortex_state.db"
```

Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Install frontend dependencies:
```bash
cd ../frontend
npm install
```

Initialize local vector database and schemas:
```bash
cd ../backend
python app/database/init_db.py
```

Launch the core system engine:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Launch the frontend (from `frontend/` directory):
```bash
npm run dev
```

## 📁 Project Structure

```
cortex/
├── backend/              # Python FastAPI backend with 22-agent swarm
│   ├── backbone_ai/      # Core AI agents and orchestration
│   ├── requirements.txt
│   └── server.py
├── frontend/             # React + TypeScript frontend dashboard
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 📝 License

Proprietary. All rights reserved.
