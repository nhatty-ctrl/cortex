from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum

# ── Enums ─────────────────────────────────────────────────────────────────────

class SignalType(str, Enum):
    BULLISH    = "bullish"
    BEARISH    = "bearish"
    NEUTRAL    = "neutral"
    WATCH      = "watch"

class ImpactLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"

class AdviceAction(str, Enum):
    BUY   = "buy"
    HOLD  = "hold"
    AVOID = "avoid"

# ── Scraped News Item ─────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    title:        str
    url:          str
    source:       str                  # e.g. "Reuters", "FT", "SEC EDGAR"
    published_at: datetime
    summary:      str
    raw_text:     Optional[str] = None
    region:       Optional[str] = None  # "US", "EU", "GLOBAL"
    tags:         List[str] = Field(default_factory=list)

# ── Impact Assessment ─────────────────────────────────────────────────────────

class CompanyImpact(BaseModel):
    ticker:       str
    company_name: str
    impact_level: ImpactLevel
    impact_reason: str
    sentiment:    SignalType
    confidence:   float = Field(ge=0.0, le=1.0)

# ── Investment Signal ─────────────────────────────────────────────────────────

class Signal(BaseModel):
    id:              str
    ticker:          str
    company_name:    str
    signal_type:     SignalType
    confidence:      float = Field(ge=0.0, le=1.0)
    headline:        str
    reasoning:       str                     # concise evidence-backed rationale
    sources:         List[str] = Field(default_factory=list)          # URLs cited
    strategy_tags:   List[str] = Field(default_factory=list)          # ["momentum", "event-driven", ...]
    risk_flags:      List[str] = Field(default_factory=list)
    created_at:      datetime = Field(default_factory=datetime.utcnow)
    affected_sector: Optional[str] = None

# ── Strategy Recommendation ───────────────────────────────────────────────────

class StrategyRecommendation(BaseModel):
    ticker:        str
    action:        AdviceAction
    time_horizon:  Literal["short", "medium", "long"]
    entry_note:    Optional[str] = None
    risk_note:     Optional[str] = None
    macro_context: Optional[str] = None
    signals_used:  List[str] = Field(default_factory=list)           # Signal IDs
    generated_at:  datetime = Field(default_factory=datetime.utcnow)

# ── Full Report ───────────────────────────────────────────────────────────────

class Report(BaseModel):
    id:              str
    ticker:          str
    company_name:    str
    generated_at:    datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str
    news_digest:     List[NewsItem] = Field(default_factory=list)
    signals:         List[Signal] = Field(default_factory=list)
    recommendation:  StrategyRecommendation
    smart_money:     Optional[dict] = None   # 13F filing data
    technical_notes: Optional[str] = None
    sources:         List[str] = Field(default_factory=list)

# ── API Request/Response schemas ──────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    ticker:    str
    company:   str
    depth:     Literal["quick", "deep"] = "quick"

class ChatMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages:     List[ChatMessage]
    ticker:       Optional[str] = None      # context — which stock are we talking about
    use_rag:      bool = True

class WatchlistItem(BaseModel):
    ticker:       str
    company_name: str
    alert_on:     List[SignalType] = Field(default_factory=lambda: [SignalType.BULLISH, SignalType.BEARISH])
