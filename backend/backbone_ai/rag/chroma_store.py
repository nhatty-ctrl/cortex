"""
Backbone AI RAG layer.

Uses ChromaDB when available and falls back to an in-memory store on environments
where native Chroma dependencies are not installed.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

from api.models.schemas import NewsItem
from config.settings import settings

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - exercised by environment
    chromadb = None
    embedding_functions = None

_client = None
_memory_docs: list[dict] = []


def _token_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def get_chroma_client():
    global _client
    if chromadb is None:
        return None
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client


def get_collection():
    client = get_chroma_client()
    if client is None:
        return None
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_news(items: List[NewsItem], ticker: str) -> int:
    collection = get_collection()
    docs, ids, metas = [], [], []

    for item in items:
        text = f"{item.title}\n\n{item.summary}"
        if item.raw_text:
            text += f"\n\n{item.raw_text[:1500]}"

        doc_id = hashlib.md5(item.url.encode()).hexdigest()
        meta = {
            "ticker": ticker,
            "source": item.source,
            "url": item.url,
            "title": item.title,
            "region": item.region or "UNKNOWN",
            "tags": ",".join(item.tags),
        }

        docs.append(text)
        ids.append(doc_id)
        metas.append(meta)
        if collection is None:
            _memory_docs.append({"id": doc_id, "text": text, "meta": meta})

    if collection is not None and docs:
        collection.upsert(documents=docs, ids=ids, metadatas=metas)

    return len(docs)


def retrieve_context(query: str, ticker: str, n_results: int = 8) -> str:
    collection = get_collection()

    if collection is not None:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"ticker": ticker},
            )
        except Exception:
            results = collection.query(query_texts=[query], n_results=n_results)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
    else:
        matches = [doc for doc in _memory_docs if doc["meta"].get("ticker") == ticker]
        if not matches:
            matches = list(_memory_docs)
        ranked = sorted(matches, key=lambda doc: _token_score(query, doc["text"]), reverse=True)[:n_results]
        docs = [doc["text"] for doc in ranked]
        metadatas = [doc["meta"] for doc in ranked]

    if not docs:
        return "No relevant context found in knowledge base."

    formatted = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "Unknown")
        url = meta.get("url", "")
        title = meta.get("title", "")
        formatted.append(f"[SOURCE: {source} | {title}]\n{doc}\nURL: {url}")
    return "\n\n---\n\n".join(formatted)


def clear_ticker(ticker: str) -> None:
    collection = get_collection()
    if collection is not None:
        collection.delete(where={"ticker": ticker})
    global _memory_docs
    _memory_docs = [doc for doc in _memory_docs if doc["meta"].get("ticker") != ticker]
