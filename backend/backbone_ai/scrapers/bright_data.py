"""
Backbone AI — Bright Data scraper layer
Covers: Reuters, FT, WSJ, Al Jazeera, SEC EDGAR, LinkedIn, SERP
Uses: Web Unlocker (JS-heavy sites) + SERP API + Scraping Browser
"""
import httpx
import asyncio
from datetime import datetime
from typing import List, Optional
from config.settings import settings
from api.models.schemas import NewsItem

# ── Bright Data base config ───────────────────────────────────────────────────

UNLOCKER_PROXY = (
    f"http://brd-customer-{settings.BRIGHT_DATA_API_KEY}"
    f"-zone-{settings.BRIGHT_DATA_UNLOCKER_ZONE}:brd_password@brd.superproxy.io:22225"
)

SERP_URL    = "https://api.brightdata.com/request"
SERP_HEADERS = {
    "Authorization": f"Bearer {settings.BRIGHT_DATA_API_KEY}",
    "Content-Type":  "application/json",
}

# ── News sources config ───────────────────────────────────────────────────────

NEWS_SOURCES = {
    "reuters":    "https://www.reuters.com/search/news?query={query}",
    "ft":         "https://www.ft.com/search?q={query}",
    "aljazeera":  "https://www.aljazeera.com/search/{query}",
    "sec_edgar":  "https://efts.sec.gov/LATEST/search-index?q=%22{query}%22&dateRange=custom&startdt={from_date}&forms=8-K,10-K,10-Q",
    "wsj":        "https://www.wsj.com/search?query={query}&mod=searchresults_viewallresults",
}

# ── SERP search (fastest — structured results) ────────────────────────────────

async def serp_search(query: str, num_results: int = 10) -> List[dict]:
    """Search Google via Bright Data SERP API and return structured results."""
    payload = {
        "zone":    settings.BRIGHT_DATA_SERP_ZONE,
        "url":     f"https://www.google.com/search?q={query}&num={num_results}",
        "format":  "json",
        "country": "us",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(SERP_URL, headers=SERP_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic", [])
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("description", ""),
                "source":  r.get("displayed_url", ""),
            }
            for r in results
        ]

# ── Web Unlocker — fetch full page text ──────────────────────────────────────

async def fetch_with_unlocker(url: str) -> str:
    """Fetch a JS-rendered or geo-blocked page via Bright Data Web Unlocker."""
    proxies = {"http://": UNLOCKER_PROXY, "https://": UNLOCKER_PROXY}
    async with httpx.AsyncClient(proxies=proxies, verify=False, timeout=60) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            return f"[fetch_error: {str(e)}]"

# ── SEC EDGAR filings ─────────────────────────────────────────────────────────

async def fetch_sec_filings(ticker: str, company: str, days_back: int = 30) -> List[NewsItem]:
    """Pull recent SEC filings (8-K, 10-K, 10-Q) for a ticker."""
    from_date = (datetime.utcnow().replace(day=1)).strftime("%Y-%m-%d")
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
        f"&dateRange=custom&startdt={from_date}&forms=8-K,10-K,10-Q"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "BackboneAI research@backbone.ai"})
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            items = []
            for h in hits[:5]:
                src = h.get("_source", {})
                items.append(NewsItem(
                    title=        src.get("file_date", "") + " — " + src.get("form_type", "Filing"),
                    url=          f"https://www.sec.gov/Archives/{src.get('file_path', '')}",
                    source=       "SEC EDGAR",
                    published_at= datetime.strptime(src.get("file_date", "2024-01-01"), "%Y-%m-%d"),
                    summary=      src.get("period_of_report", ""),
                    tags=         [src.get("form_type", ""), ticker],
                    region=       "US",
                ))
            return items
        except Exception as e:
            return []

# ── News scrape — ticker-aware ────────────────────────────────────────────────

async def scrape_news_for_ticker(ticker: str, company: str, depth: str = "quick") -> List[NewsItem]:
    """
    Main news ingestion function.
    quick  → SERP only (fast, ~5s)
    deep   → SERP + full page fetch from top 3 sources (~20s)
    """
    queries = [
        f"{company} stock news",
        f"{ticker} earnings analyst",
        f"{company} CEO insider SEC filing",
        f"{company} market impact geopolitical",
    ]

    all_results: List[dict] = []

    # Run all SERP queries in parallel
    tasks = [serp_search(q, num_results=5) for q in queries]
    batches = await asyncio.gather(*tasks, return_exceptions=True)
    for batch in batches:
        if isinstance(batch, list):
            all_results.extend(batch)

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    items: List[NewsItem] = []

    if depth == "deep":
        # Fetch full text from top 5 unique results using Web Unlocker
        fetch_tasks = [fetch_with_unlocker(r["url"]) for r in unique[:5]]
        full_texts  = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for r, text in zip(unique[:5], full_texts):
            raw = text if isinstance(text, str) else ""
            items.append(NewsItem(
                title=        r["title"],
                url=          r["url"],
                source=       r.get("source", "Web"),
                published_at= datetime.utcnow(),
                summary=      r["snippet"],
                raw_text=     raw[:3000],     # cap at 3k chars per doc
                tags=         [ticker, company],
            ))
    else:
        for r in unique[:10]:
            items.append(NewsItem(
                title=        r["title"],
                url=          r["url"],
                source=       r.get("source", "Web"),
                published_at= datetime.utcnow(),
                summary=      r["snippet"],
                tags=         [ticker, company],
            ))

    # Always append SEC filings
    sec_items = await fetch_sec_filings(ticker, company)
    items.extend(sec_items)

    return items

# ── LinkedIn executive activity ───────────────────────────────────────────────

async def scrape_linkedin_execs(company: str) -> List[dict]:
    """Search for executive moves, layoffs, and headcount signals for a company."""
    queries = [
        f"{company} CEO joins leaves linkedin",
        f"{company} layoffs hiring headcount 2025",
        f"{company} executive appointment site:linkedin.com OR site:businesswire.com",
    ]
    tasks   = [serp_search(q, num_results=3) for q in queries]
    batches = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for batch in batches:
        if isinstance(batch, list):
            results.extend(batch)
    return results

# ── Smart Money — 13F filings ─────────────────────────────────────────────────

async def fetch_smart_money(ticker: str) -> dict:
    """
    Scrape 13F SEC filings to see if Buffett/Soros/Ackman hold this ticker.
    Uses WhaleWisdom via Web Unlocker (JS-heavy site).
    """
    url  = f"https://whalewisdom.com/stock/{ticker.lower()}"
    html = await fetch_with_unlocker(url)
    # Return raw HTML — Gemini will parse and extract the meaningful parts
    return {"ticker": ticker, "raw_html": html[:4000], "source": "whalewisdom.com"}

# ── World/macro news ─────────────────────────────────────────────────────────

async def fetch_macro_news() -> List[NewsItem]:
    """Pull macro / geopolitical news that could affect markets broadly."""
    queries = [
        "Federal Reserve interest rate decision today",
        "global market crash recession 2025",
        "geopolitical risk oil supply disruption",
        "central bank ECB BOJ policy announcement",
        "China US trade tariff news",
    ]
    tasks   = [serp_search(q, num_results=3) for q in queries]
    batches = await asyncio.gather(*tasks, return_exceptions=True)
    items   = []
    for batch in batches:
        if isinstance(batch, list):
            for r in batch:
                items.append(NewsItem(
                    title=        r["title"],
                    url=          r["url"],
                    source=       r.get("source", "Web"),
                    published_at= datetime.utcnow(),
                    summary=      r["snippet"],
                    tags=         ["macro", "global"],
                    region=       "GLOBAL",
                ))
    return items
