"""Watchlist router backed by the live engine's shared runtime state."""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from api.routers.live_engine import add_watch_item, get_watchlist, remove_watch_item

router = APIRouter()


class WatchItem(BaseModel):
    ticker: str
    company: str


@router.get("/", response_model=List[WatchItem])
async def get():
    return get_watchlist()


@router.post("/", response_model=WatchItem)
async def add(item: WatchItem):
    return add_watch_item(item.ticker, item.company)


@router.delete("/{ticker}")
async def remove(ticker: str):
    deleted = remove_watch_item(ticker)
    return {"deleted": ticker.upper(), "found": deleted}
