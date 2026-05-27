from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # ── Bright Data ───────────────────────────────────────────────
    BRIGHT_DATA_API_KEY:        str = ""
    BRIGHT_DATA_SERP_ZONE:      str = "serp_api1"
    BRIGHT_DATA_UNLOCKER_ZONE:  str = "web_unlocker1"
    BRIGHT_DATA_SCRAPER_ZONE:   str = "datacenter1"
    BRIGHT_DATA_MCP_URL:        str = "https://mcp.brightdata.com/sse"
    # Browser API (Scraping Browser / CDP)
    BRIGHT_DATA_CUSTOMER_ID:    str = ""   # brd-customer-XXXXXXXX
    BRIGHT_DATA_BROWSER_ZONE:   str = "scraping_browser1"
    BRIGHT_DATA_BROWSER_PASSWORD: str = ""
    # Dataset API IDs (pre-built scrapers)
    BD_DATASET_LINKEDIN:        str = "gd_lh9q6t50qlaguy3aj"
    BD_DATASET_SEC:             str = "gd_m8sn95m43q5q0"
    BD_DATASET_REDDIT:          str = "gd_l7q7dkf244hwjntr0"

    # ── Google Gemini ─────────────────────────────────────────────
    GEMINI_API_KEY:    str = ""
    GEMINI_MODEL:      str = "gemini-2.0-flash"
    GEMINI_PRO_MODEL:  str = "gemini-2.5-pro"

    # ── DeepSeek ──────────────────────────────────────────────────
    DEEPSEEK_API_KEY:  str = ""
    DEEPSEEK_MODEL:    str = "deepseek-reasoner"

    # ── ChromaDB ──────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION:  str = "cortex_finance"

    # ── App ───────────────────────────────────────────────────────
    DEBUG:             bool = True
    ENABLE_SCHEDULER:  bool = True
    SCRAPE_INTERVAL_S: int  = 300   # 5 minutes
    SIGNAL_CACHE_TTL:  int  = 300
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_allow_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.GEMINI_API_KEY or self.DEEPSEEK_API_KEY)

    @property
    def has_scraping_credentials(self) -> bool:
        return bool(self.BRIGHT_DATA_API_KEY)

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
