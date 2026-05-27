"""
Cortex AI main entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.chat import router as chat_router
from api.routers.live_engine import (
    get_scheduler_status,
    router as live_router,
    start_scheduler,
    stop_scheduler,
)
from api.routers.reports import router as reports_router
from api.routers.scrape import router as scrape_router
from api.routers.signals import router as signals_router
from api.routers.watchlist import router as watchlist_router
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if settings.ENABLE_SCHEDULER:
        task = await start_scheduler()
    app.state.scheduler_task = task
    yield
    await stop_scheduler()


app = FastAPI(
    title="Cortex AI - Goldman-Grade Finance Intelligence",
    version="1.0.0",
    description=(
        "22-agent real-time financial intelligence platform. "
        "Bright Data scraping + Gemini + DeepSeek + ChromaDB RAG. "
        "Track 2: Finance & Market Intelligence."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals_router, prefix="/api/signals", tags=["Signals"])
app.include_router(live_router, prefix="/api/signals", tags=["Live Engine"])
app.include_router(reports_router, prefix="/api/reports", tags=["Reports"])
app.include_router(scrape_router, prefix="/api/scrape", tags=["Scrape"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(watchlist_router, prefix="/api/watchlist", tags=["Watchlist"])


@app.get("/")
def root():
    scheduler_status = get_scheduler_status()
    return {
        "app": "Cortex AI",
        "version": "1.0.0",
        "status": "live",
        "agents": 22,
        "scheduler": scheduler_status["status"],
        "docs": "/docs",
        "track": "Finance & Market Intelligence - Web Data UNLOCKED Hackathon",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "scheduler": get_scheduler_status(),
        "llm_configured": settings.has_llm_credentials,
        "scraping_configured": settings.has_scraping_credentials,
    }
