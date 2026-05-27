"""Cortex AI — Chat router (RAG-grounded)"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag.cortex_rag import retrieve, format_context
from config.settings import settings
import google.generativeai as genai

genai.configure(api_key=settings.GEMINI_API_KEY)
_flash = genai.GenerativeModel(settings.GEMINI_MODEL)

router = APIRouter()

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: list[ChatMsg]
    ticker:   Optional[str] = None

@router.post("/")
async def chat(req: ChatReq):
    try:
        last = next((m.content for m in reversed(req.messages) if m.role=="user"), "")
        context = ""
        if req.ticker:
            chunks  = retrieve(last, req.ticker, n=8)
            context = format_context(chunks)
        history = "\n".join([f"{m.role.upper()}: {m.content}" for m in req.messages[:-1]])
        prompt = f"""You are Cortex AI — an institutional-grade financial analyst.
Answer using ONLY the context below. Cite sources. Be specific and direct.

CONTEXT:
{context or 'No specific data loaded.'}

HISTORY:
{history}

USER: {last}

ANSWER:"""
        resp = _flash.generate_content(prompt)
        return {"role": "assistant", "content": resp.text}
    except Exception as e:
        raise HTTPException(500, str(e))
