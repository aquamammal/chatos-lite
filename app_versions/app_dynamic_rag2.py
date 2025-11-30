import os
import json
from typing import List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# ------------------------------------------------
# Paths / FastAPI setup
# ------------------------------------------------

# This file lives in app_versions/, so BASE_DIR is one level up
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

app = FastAPI()

# Serve static frontend from ../static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


# ------------------------------------------------
# Request / response models
# ------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    # model is optional; if not provided we default to qwen2.5:7b
    model: Optional[str] = None
    use_rag: bool = False


class ChatResponse(BaseModel):
    answer: str
    model: str


# ------------------------------------------------
# RAG-lite: TXT + EPUB
# ------------------------------------------------

DOCS_TEXT: str = ""  # combined context string from all docs


def extract_epub_text(path: str) -> str:
    """
    Extract text from an EPUB using ebooklib + BeautifulSoup.
    """
    print(f"[RAG] Extracting EPUB: {path}")
    try:
        book = epub.read_epub(path)
    except Exception as e:
        print(f"[RAG] Failed to read EPUB {path}: {e}")
        return ""

    parts: List[str] = []
    try:
        items = list(book.get_items())
        print(f"[RAG] EPUB {path} has {len(items)} items")
        for item in items:
            try:
                if item.get_type() == ITEM_DOCUMENT:
                    html = item.get_body_content()
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator="\n")
                    if text.strip():
                        parts.append(text.strip())
            except Exception as e:
                print(f"[RAG] Error processing item in {path}: {e}")
                continue
    except Exception as e:
        print(f"[RAG] Unexpected error iterating items in {path}: {e}")
        return ""

    final = "\n\n".join(parts).strip()
    print(f"[RAG] EPUB {path} extracted length: {len(final)} chars")
    return final


def load_docs_text() -> None:
    """
    Read all .txt and .epub files from ../docs and concatenate them
    into one big context string. Rescans the folder each time so
    new files are picked up automatically.
    """
    global DOCS_TEXT

    docs_dir = os.path.abspath(DOCS_DIR)

    if not os.path.isdir(docs_dir):
        print(f"[RAG] No docs directory at {docs_dir}, skipping.")
        DOCS_TEXT = ""
        return

    print(f"[RAG] Scanning docs in {docs_dir} ...")
    parts: List[str] = []

    for fname in os.listdir(docs_dir):
        full_path = os.path.join(docs_dir, fname)
        if not os.path.isfile(full_path):
            continue

        lower = fname.lower()
        text = ""

        if lower.endswith(".txt"):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception as e:
                print(f"[RAG] Error reading TXT {full_path}: {e}")
                continue

        elif lower.endswith(".epub"):
            text = extract_epub_text(full_path)

        else:
            # ignore other file types
            continue

        if not text.strip():
            continue

        parts.append(f"[{fname}]\n{text}")
        print(f"[RAG] Loaded {fname}")

    DOCS_TEXT = "\n\n".join(parts)
    print(f"[RAG] Total docs size: {len(DOCS_TEXT)} chars")


def ensure_docs_text_loaded() -> None:
    """
    Always reload docs on each RAG call so new files are picked up.
    """
    load_docs_text()


def get_personal_notes() -> str:
    """
    Read docs/notes.txt if it exists.
    Use this for short, high-priority info like your name,
    what you're building, etc.
    """
    notes_path = os.path.join(DOCS_DIR, "notes.txt")
    try:
        if os.path.isfile(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        print(f"[RAG] Error reading personal notes: {e}")
    return ""


def maybe_add_rag_context(messages: List[dict], use_rag: bool) -> List[dict]:
    """
    If use_rag is True, prepend a system message with:
      - PERSONAL NOTES (from notes.txt) first
      - Then a truncated version of all other docs (DOCS_TEXT)
    """
    if not use_rag:
        return messages

    ensure_docs_text_loaded()

    personal = get_personal_notes()
    docs_text = DOCS_TEXT

    # Limit huge book text so we don't blow up the prompt
    max_docs_chars = 60000
    if len(docs_text) > max_docs_chars:
        docs_text = docs_text[:max_docs_chars]
        print(f"[RAG] Truncated docs context to {max_docs_chars} chars")

    if not personal and not docs_text:
        return messages

    parts: List[str] = [
        "You are an assistant with access to my personal notes and documents.",
        "Use the PERSONAL NOTES with highest priority when answering questions "
        "about me (my name, what I'm building, etc.).",
        "Use the OTHER CONTEXT only when it's relevant to the user's question.",
        "",
    ]

    if personal:
        parts.append("PERSONAL NOTES:")
        parts.append(personal)
        parts.append("")

    if docs_text:
        parts.append("OTHER CONTEXT:")
        parts.append(docs_text)
        parts.append("")

    system_content = "\n".join(parts)

    return [{"role": "system", "content": system_content}] + messages


def get_model_name(req: ChatRequest) -> str:
    """
    Determine which model to use.
    - If req.model is set (from the UI), use that
    - Otherwise fall back to qwen2.5:7b
    """
    return req.model or "qwen2.5:7b"


# ------------------------------------------------
# Non-streaming chat (JSON)
# ------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    base_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages_for_model = maybe_add_rag_context(base_messages, req.use_rag)

    model_name = get_model_name(req)
    ollama_payload = {
        "model": model_name,
        "messages": messages_for_model,
        "stream": False,
    }

    resp = requests.post("http://localhost:11434/api/chat", json=ollama_payload)
    resp.raise_for_status()
    data = resp.json()

    answer = data["message"]["content"]
    return ChatResponse(answer=answer, model=model_name)


# ------------------------------------------------
# Streaming chat with RAG-lite
# ------------------------------------------------

@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):
    """
    Streaming endpoint: optionally prepends docs text as a system message,
    then forwards chunks from Ollama as plain text.
    """
    base_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages_for_model = maybe_add_rag_context(base_messages, req.use_rag)

    model_name = get_model_name(req)

    def generate():
        ollama_payload = {
            "model": model_name,
            "messages": messages_for_model,
            "stream": True,
        }

        with requests.post(
            "http://localhost:11434/api/chat",
            json=ollama_payload,
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                msg = data.get("message", {})
                chunk = msg.get("content", "")
                if chunk:
                    yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
