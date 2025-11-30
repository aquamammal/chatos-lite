import os
import json
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# ------------------------------------------------
# Setup paths
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ------------------------------------------------
# Request / response models
# ------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    use_rag: bool = False

class ChatResponse(BaseModel):
    answer: str
    model: str


# ------------------------------------------------
# FULL EPUB/TXT LOADER (NO TRUNCATION)
# ------------------------------------------------

FULL_DOC_TEXT: str = ""
NOTES_TEXT: str = ""


def extract_epub_text(path: str) -> str:
    print(f"[RAG] Extracting EPUB: {path}")
    try:
        book = epub.read_epub(path)
    except Exception as e:
        print(f"[RAG] EPUB read error: {e}")
        return ""

    parts = []
    for item in book.get_items():
        try:
            if item.get_type() == ITEM_DOCUMENT:
                html = item.get_body_content()
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    parts.append(text)
        except Exception:
            continue

    combined = "\n\n".join(parts).strip()
    print(f"[RAG] EPUB extracted {len(combined)} chars")
    return combined


def load_all_text():
    """
    Load ALL .txt and .epub files.
    NO LIMITS. NO TRUNCATION.
    """
    global FULL_DOC_TEXT, NOTES_TEXT

    ALL = []

    # Load personal notes
    notes_path = os.path.join(DOCS_DIR, "notes.txt")
    if os.path.isfile(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            NOTES_TEXT = f.read().strip()
            print(f"[RAG] Loaded notes.txt ({len(NOTES_TEXT)} chars)")

    # Load all docs
    total_chars = 0

    for fname in os.listdir(DOCS_DIR):
        full_path = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(full_path):
            continue

        lower = fname.lower()
        text = ""

        if lower.endswith(".txt"):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except:
                continue

        elif lower.endswith(".epub"):
            text = extract_epub_text(full_path)

        else:
            continue

        if text.strip():
            ALL.append(f"[{fname}]\n{text}")
            total_chars += len(text)

    FULL_DOC_TEXT = "\n\n".join(ALL)
    print(f"[RAG] TOTAL TEXT SIZE (NO LIMIT): {len(FULL_DOC_TEXT)} chars")


def ensure_loaded():
    # Reload every request so you can update docs
    load_all_text()


# ------------------------------------------------
# Build system message (dump everything)
# ------------------------------------------------

def build_full_dump_context() -> Optional[str]:
    """
    NO LIMITS.
    Dump every character of all documents into the system prompt.
    """
    ensure_loaded()

    global FULL_DOC_TEXT, NOTES_TEXT

    if not FULL_DOC_TEXT and not NOTES_TEXT:
        return None

    parts = [
        "You are an assistant with access to my ENTIRE document library.",
        "Below is ALL document text, provided IN FULL with NO TRUNCATION.",
        "Use it when relevant to the user's query.",
        "",
    ]

    if NOTES_TEXT:
        parts.append("PERSONAL NOTES:")
        parts.append(NOTES_TEXT)
        parts.append("")

    if FULL_DOC_TEXT:
        parts.append("FULL DOCUMENT DUMP:")
        parts.append(FULL_DOC_TEXT)
        parts.append("")

    combined = "\n".join(parts)
    print(f"[RAG] FULL SYSTEM MESSAGE SIZE = {len(combined)} chars (NO LIMITS)")

    return combined


def maybe_add_rag_context(messages: List[Dict[str, Any]], use_rag: bool):
    if not use_rag:
        return messages

    system_content = build_full_dump_context()
    if not system_content:
        return messages

    return [{"role": "system", "content": system_content}] + messages


def get_model_name(req: ChatRequest) -> str:
    return req.model or "qwen2.5:7b"


# ------------------------------------------------
# Chat endpoints
# ------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = maybe_add_rag_context(messages, req.use_rag)
    model = get_model_name(req)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    r = requests.post("http://localhost:11434/api/chat", json=payload)
    r.raise_for_status()
    data = r.json()

    return ChatResponse(answer=data["message"]["content"], model=model)


@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = maybe_add_rag_context(messages, req.use_rag)
    model = get_model_name(req)

    def generate():
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        with requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except:
                    continue

                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


# ------------------------------------------------
# Run directly
# ------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_dump_all_no_limit:app", host="127.0.0.1", port=8000, reload=True)
