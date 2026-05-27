"""
Cortex AI — Bright Data Client
================================
Implements all 4 Bright Data tools exactly as documented:

  1. SERP API        — search_engine tool via proxy port 33335
                       POST https://api.brightdata.com/serp/req
  2. Web Unlocker    — unblock_url via proxy brd.superproxy.io:33335
                       or API POST https://api.brightdata.com/request
  3. Scraping Browser— Playwright via CDP endpoint
                       wss://brd-customer-{id}-zone-{zone}:...@brd.superproxy.io:9222
  4. Web Scraper API — dataset snapshots + trigger
                       POST https://api.brightdata.com/datasets/v3/trigger

Auth: Bearer token in Authorization header.
Charged only on successful requests.
"""

import asyncio
import json
import re
import time
import httpx
from typing import Optional
from config.settings import settings

# ── Bright Data endpoints (from docs) ────────────────────────────────────────
BD_SERP_PROXY   = "brd.superproxy.io:33335"
BD_SERP_API_URL = "https://api.brightdata.com/serp/req"
BD_UNLOCKER_URL = "https://api.brightdata.com/request"
BD_DATASET_URL  = "https://api.brightdata.com/datasets/v3"
BD_BASE_HEADERS = {
    "Authorization": f"Bearer {settings.BRIGHT_DATA_API_KEY}",
    "Content-Type":  "application/json",
}

# ── Emit hook — filled by live_engine.py at startup ──────────────────────────
_emit_scrape_hook = None

def set_emit_hook(fn):
    global _emit_scrape_hook
    _emit_scrape_hook = fn

async def _emit(msg: str, source: str = "", status: str = "running"):
    if _emit_scrape_hook:
        await _emit_scrape_hook(msg, source, status)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — SERP API
# POST to BD_SERP_API_URL with zone + url + format: json
# Returns structured organic results, news, prices
# Use for: Reuters, FT, Google News, SEC EDGAR search, any search query
# ══════════════════════════════════════════════════════════════════════════════

async def serp_search(
    query:   str,
    engine:  str = "google",          # google | bing | yandex | duckduckgo
    country: str = "us",
    num:     int = 10,
    news:    bool = False,
) -> list[dict]:
    """
    Real-time structured search via Bright Data SERP API.
    Returns list of {title, url, snippet, source} dicts.
    Charged only on success.
    """
    await _emit(f"SERP search: {query[:50]}…", source="Bright Data SERP API")

    # Build the target URL — append tbm=nws for news mode
    tbm   = "&tbm=nws" if news else ""
    q_enc = query.replace(" ", "+")
    target_url = (
        f"https://www.google.com/search?q={q_enc}&num={num}&gl={country}{tbm}"
        if engine == "google"
        else f"https://www.bing.com/search?q={q_enc}&count={num}"
    )

    payload = {
        "zone":    settings.BRIGHT_DATA_SERP_ZONE,
        "url":     target_url,
        "format":  "json",
        "country": country,
    }

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            r = await client.post(
                BD_SERP_API_URL,
                headers=BD_BASE_HEADERS,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

            # Normalise: organic results
            results = []
            for item in data.get("organic", [])[:num]:
                results.append({
                    "title":   item.get("title", ""),
                    "url":     item.get("url",   ""),
                    "snippet": item.get("description", ""),
                    "source":  item.get("displayed_url", engine),
                })
            # Also surface news results if present
            for item in data.get("news", [])[:5]:
                results.append({
                    "title":   item.get("title", ""),
                    "url":     item.get("url",   ""),
                    "snippet": item.get("description", ""),
                    "source":  item.get("source", "news"),
                    "published": item.get("date", ""),
                })
            await _emit(f"✓ SERP: {len(results)} results for '{query[:30]}'",
                        source="Bright Data SERP API", status="done")
            return results

        except httpx.HTTPStatusError as e:
            await _emit(f"SERP error {e.response.status_code}: {query[:30]}",
                        source="Bright Data SERP API", status="error")
            return []
        except Exception as e:
            await _emit(f"SERP exception: {str(e)[:60]}",
                        source="Bright Data SERP API", status="error")
            return []


async def multi_serp(queries: list[str], **kwargs) -> list[dict]:
    """Run multiple SERP queries in parallel. Deduplicates by URL."""
    tasks  = [serp_search(q, **kwargs) for q in queries]
    batches= await asyncio.gather(*tasks, return_exceptions=True)
    seen, out = set(), []
    for batch in batches:
        if isinstance(batch, list):
            for r in batch:
                if r["url"] not in seen:
                    seen.add(r["url"])
                    out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — WEB UNLOCKER API
# POST to BD_UNLOCKER_URL with zone + url + format
# Handles CAPTCHA, fingerprinting, JS rendering, retries automatically
# Use for: LinkedIn, WhaleWisdom, Reuters full articles, WSJ, Bloomberg
# ══════════════════════════════════════════════════════════════════════════════

async def unlocker_fetch(
    url:     str,
    fmt:     str  = "markdown",    # markdown | raw | screenshot
    country: str  = "us",
    render:  bool = True,          # JS rendering
) -> str:
    """
    Fetch any URL bypassing bot detection via Bright Data Web Unlocker.
    Returns markdown by default (LLM-ready).
    Only charged on success.
    """
    await _emit(f"Unlocking: {url[:60]}…", source="Bright Data Web Unlocker")

    payload = {
        "zone":    settings.BRIGHT_DATA_UNLOCKER_ZONE,
        "url":     url,
        "format":  fmt,
        "country": country,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                BD_UNLOCKER_URL,
                headers=BD_BASE_HEADERS,
                json=payload,
            )
            r.raise_for_status()
            content = r.text
            await _emit(f"✓ Unlocked: {url[:50]} ({len(content)} chars)",
                        source="Bright Data Web Unlocker", status="done")
            return content[:8000]   # cap at 8k chars per page

        except httpx.HTTPStatusError as e:
            await _emit(f"Unlocker {e.response.status_code}: {url[:40]}",
                        source="Bright Data Web Unlocker", status="error")
            return f"[unlocker_error_{e.response.status_code}]"
        except Exception as e:
            await _emit(f"Unlocker exception: {str(e)[:60]}",
                        source="Bright Data Web Unlocker", status="error")
            return f"[unlocker_exception]"


async def unlocker_fetch_many(urls: list[str], **kwargs) -> list[str]:
    """Fetch multiple URLs in parallel with Web Unlocker."""
    tasks = [unlocker_fetch(u, **kwargs) for u in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — SCRAPING BROWSER (Browser API)
# CDP endpoint: wss://brd-customer-{id}-zone-{zone}@brd.superproxy.io:9222
# Use for: X.com (Twitter), LinkedIn profile pages, Reddit (JS-heavy)
# Playwright-based. Full browser automation.
# ══════════════════════════════════════════════════════════════════════════════

def get_browser_cdp_url() -> str:
    """Build the CDP WebSocket URL for Bright Data's Browser API."""
    customer = settings.BRIGHT_DATA_CUSTOMER_ID      # from .env
    zone     = settings.BRIGHT_DATA_BROWSER_ZONE     # from .env
    password = settings.BRIGHT_DATA_BROWSER_PASSWORD # from .env
    return f"wss://brd-customer-{customer}-zone-{zone}:{password}@brd.superproxy.io:9222"


async def browser_fetch(url: str, wait_selector: str = "body") -> str:
    """
    Fetch a JS-heavy page via Bright Data Browser API (Playwright CDP).
    Falls back to Web Unlocker if playwright not installed.

    Install: pip install playwright && playwright install chromium
    """
    await _emit(f"Browser fetch: {url[:60]}…", source="Bright Data Browser API")

    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(get_browser_cdp_url())
            page    = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_selector(wait_selector, timeout=15000)
            except Exception:
                pass
            content = await page.inner_text("body")
            await browser.close()
            await _emit(f"✓ Browser: {url[:50]} ({len(content)} chars)",
                        source="Bright Data Browser API", status="done")
            return content[:6000]

    except ImportError:
        # Playwright not installed — fall back to Web Unlocker
        await _emit("Playwright not installed — falling back to Web Unlocker",
                    source="Browser API fallback", status="running")
        return await unlocker_fetch(url, render=True)

    except Exception as e:
        await _emit(f"Browser error: {str(e)[:60]}",
                    source="Bright Data Browser API", status="error")
        return await unlocker_fetch(url, render=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — WEB SCRAPER API (Dataset API)
# Pre-built scrapers for LinkedIn, Amazon, SEC, Reddit, etc.
# POST trigger → GET snapshot
# Use for: SEC EDGAR structured data, LinkedIn company profiles
# ══════════════════════════════════════════════════════════════════════════════

async def dataset_trigger(
    dataset_id: str,
    inputs:     list[dict],
) -> Optional[str]:
    """
    Trigger a Bright Data dataset scrape job.
    Returns snapshot_id or None on failure.

    Common dataset IDs:
      gd_lh9q6t50qlaguy3aj  — LinkedIn company profiles
      gd_m8sn95m43q5q0       — SEC EDGAR filings
      gd_l7q7dkf244hwjntr0  — Reddit posts
    """
    await _emit(f"Dataset trigger: {dataset_id}", source="Bright Data Dataset API")

    payload = {"inputs": inputs}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                f"{BD_DATASET_URL}/trigger?dataset_id={dataset_id}&format=json",
                headers=BD_BASE_HEADERS,
                json=payload,
            )
            r.raise_for_status()
            snapshot_id = r.json().get("snapshot_id")
            await _emit(f"✓ Dataset job started: {snapshot_id}",
                        source="Bright Data Dataset API", status="done")
            return snapshot_id
        except Exception as e:
            await _emit(f"Dataset trigger error: {str(e)[:60]}",
                        source="Bright Data Dataset API", status="error")
            return None


async def dataset_get(snapshot_id: str, timeout_s: int = 60) -> list[dict]:
    """Poll for dataset snapshot completion and return results."""
    await _emit(f"Polling dataset: {snapshot_id}", source="Bright Data Dataset API")
    deadline = time.monotonic() + timeout_s

    async with httpx.AsyncClient(timeout=30) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(
                    f"{BD_DATASET_URL}/snapshot/{snapshot_id}?format=json",
                    headers=BD_BASE_HEADERS,
                )
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        await _emit(f"✓ Dataset ready: {len(data)} records",
                                    source="Bright Data Dataset API", status="done")
                        return data
                await asyncio.sleep(5)
            except Exception:
                await asyncio.sleep(5)
    return []


# ══════════════════════════════════════════════════════════════════════════════
# HIGHER-LEVEL FINANCE SCRAPERS
# Built on top of the 4 tools above
# ══════════════════════════════════════════════════════════════════════════════

async def scrape_news(ticker: str, company: str, depth: str = "quick") -> list[dict]:
    """
    Multi-query news scrape for a ticker.
    quick = SERP only (4 queries, ~5s)
    deep  = SERP + full article fetch via Unlocker (~20s)
    """
    queries = [
        f"{company} stock news today",
        f"{ticker} earnings SEC filing analyst",
        f"{company} CEO CFO insider executive",
        f"{company} geopolitical macro impact",
    ]
    results = await multi_serp(queries, news=True)

    if depth == "deep" and results:
        top_urls = [r["url"] for r in results[:5] if r.get("url")]
        full_texts = await unlocker_fetch_many(top_urls)
        for i, text in enumerate(full_texts[:5]):
            if i < len(results) and isinstance(text, str):
                results[i]["full_text"] = text[:2000]

    return results


async def scrape_sec_filings(ticker: str) -> list[dict]:
    """
    Scrape SEC EDGAR for recent filings: 8-K, 10-K, 10-Q, 13F, Form 4.
    Uses SERP API to find EDGAR URLs, then Unlocker to fetch content.
    """
    await _emit(f"Checking SEC EDGAR for {ticker}…", source="SEC EDGAR")

    queries = [
        f"site:sec.gov {ticker} 8-K filing 2025",
        f"site:sec.gov {ticker} 10-Q 10-K annual report",
        f"site:sec.gov {ticker} Form 4 insider transaction",
        f"site:sec.gov {ticker} 13F institutional holdings",
    ]
    serp_results = await multi_serp(queries)
    sec_urls = [r["url"] for r in serp_results if "sec.gov" in r.get("url", "")][:6]

    filings = []
    if sec_urls:
        texts = await unlocker_fetch_many(sec_urls[:4])
        for url, text in zip(sec_urls, texts):
            if isinstance(text, str) and len(text) > 100:
                filings.append({"url": url, "content": text, "source": "SEC EDGAR"})

    await _emit(f"✓ SEC: {len(filings)} filings for {ticker}",
                source="SEC EDGAR", status="done")
    return filings


async def scrape_linkedin_exec(company: str) -> list[dict]:
    """
    Detect C-suite changes via Bright Data Browser API (JS-heavy LinkedIn).
    Falls back to SERP + Unlocker if browser unavailable.
    """
    await _emit(f"Scanning LinkedIn execs: {company}…", source="LinkedIn / Browser API")

    # SERP approach for executive news
    queries = [
        f"{company} CEO CFO COO resigned departed left site:linkedin.com OR site:businesswire.com",
        f"{company} new CEO appointed hired joins 2025",
        f"{company} executive departure announcement",
    ]
    results = await multi_serp(queries)

    # Try browser fetch for LinkedIn company page
    company_slug = company.lower().replace(" ", "-").replace(".", "")
    li_url = f"https://www.linkedin.com/company/{company_slug}/people/"
    li_content = await browser_fetch(li_url, wait_selector=".org-people")

    if li_content and len(li_content) > 200:
        results.append({
            "title":   f"LinkedIn: {company} people page",
            "url":     li_url,
            "snippet": li_content[:500],
            "source":  "LinkedIn",
        })

    await _emit(f"✓ LinkedIn: {len(results)} exec signals for {company}",
                source="LinkedIn", status="done")
    return results


async def scrape_x_accounts(tickers: list[str], companies: list[str]) -> list[dict]:
    """
    Scrape high-signal X/Twitter accounts via Bright Data Browser API.
    Tier 1: @elonmusk, @chamath, @BillAckman, @carlicahn, @naval
    Tier 2: @markets, @zerohedge, @FT, @WSJmarkets
    """
    await _emit("Scanning X.com high-signal accounts…", source="X.com / Browser API")

    tier1 = ["elonmusk", "chamath", "BillAckman", "naval", "carlicahn"]
    posts = []

    # SERP approach (faster than browser for initial scan)
    keywords = " OR ".join(tickers + companies)
    queries  = [
        f"site:x.com ({keywords}) stock invest",
        f"site:twitter.com ({keywords}) bullish bearish",
    ]
    for account in tier1[:3]:  # limit to 3 for speed
        queries.append(f"site:x.com/{account} ({keywords})")

    results = await multi_serp(queries)
    for r in results:
        posts.append({
            "account": r.get("source", "x.com"),
            "title":   r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "url":     r.get("url", ""),
            "tier":    1 if any(t in r.get("url","") for t in tier1) else 2,
        })

    await _emit(f"✓ X.com: {len(posts)} social signals",
                source="X.com / Browser API", status="done")
    return posts


async def scrape_smart_money(ticker: str) -> dict:
    """
    Scrape 13F smart money positioning via Web Unlocker.
    Sources: WhaleWisdom, SEC EDGAR 13F search.
    """
    await _emit(f"Smart money 13F: {ticker}…", source="WhaleWisdom / SEC")

    ww_url = f"https://whalewisdom.com/stock/{ticker.lower()}"
    content = await unlocker_fetch(ww_url, fmt="markdown")

    # Also search SEC for 13F filings mentioning this ticker
    sec_results = await serp_search(
        f"site:sec.gov 13F {ticker} institutional holdings",
        num=5
    )

    await _emit(f"✓ Smart money data for {ticker}",
                source="WhaleWisdom", status="done")
    return {
        "ticker":      ticker,
        "whalewisdom": content[:3000],
        "sec_13f":     sec_results[:3],
        "source":      "WhaleWisdom + SEC EDGAR via Bright Data",
    }


async def scrape_macro_news() -> list[dict]:
    """
    Global macro news: Fed, ECB, OPEC, geopolitical, commodity prices.
    Runs once per cycle, shared across all ticker analyses.
    """
    await _emit("Fetching global macro intelligence…", source="Reuters / AJZ / Fed")

    queries = [
        "Federal Reserve interest rate decision statement",
        "ECB European Central Bank monetary policy",
        "OPEC oil production cut decision",
        "geopolitical risk oil supply disruption sanctions",
        "China US trade tariff policy 2025",
        "inflation CPI PCE data release",
        "global recession risk GDP outlook",
        "gold silver price drivers today",
        "cryptocurrency bitcoin market",
    ]
    results = await multi_serp(queries, news=True, num=5)

    # Fetch Al Jazeera and BBC for geopolitical depth
    geo_urls = [
        "https://www.aljazeera.com/economy/",
        "https://www.bbc.com/news/business",
    ]
    for url in geo_urls:
        content = await unlocker_fetch(url, fmt="markdown")
        if len(content) > 200:
            results.append({
                "title":   f"Live: {url.split('/')[2]}",
                "url":     url,
                "snippet": content[:600],
                "source":  url.split("/")[2],
                "region":  "GLOBAL",
            })

    await _emit(f"✓ Macro: {len(results)} signals",
                source="Global macro", status="done")
    return results


async def scrape_historical_event(event_query: str) -> str:
    """
    Browse the web for historical market event data.
    Used by HistoricalPlaybook agent — NOT pre-stored, always live.
    e.g. "2020 COVID stock market crash what happened recovery"
    """
    await _emit(f"Browsing history: {event_query[:50]}…", source="Bright Data SERP")

    results = await multi_serp(
        [event_query, f"{event_query} investor playbook what worked"],
        news=False, num=8
    )

    # Fetch top 2 articles for depth
    top_urls = [r["url"] for r in results[:2] if r.get("url")]
    if top_urls:
        texts = await unlocker_fetch_many(top_urls)
        for i, text in enumerate(texts):
            if isinstance(text, str) and i < len(results):
                results[i]["full_text"] = text[:2000]

    combined = "\n\n".join([
        f"[{r.get('source','')}] {r.get('title','')}\n"
        f"{r.get('snippet','')}\n"
        f"{r.get('full_text','')[:500]}"
        for r in results[:5]
    ])

    await _emit(f"✓ Historical data fetched for: {event_query[:30]}",
                source="Bright Data SERP", status="done")
    return combined
