import os
import json
from typing import List

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# ------------------------------------------------
# FastAPI app setup
# ------------------------------------------------

app = FastAPI()

# Serve static frontend from ../static
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go up from app_versions/ to project root
STATIC_DIR = os.path.join(BASE_DIR, "static")

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
    model: str = "qwen2.5:7b"  # default chat model
    use_rag: bool = False      # whether to inject docs as context


class ChatResponse(BaseModel):
    answer: str
    model: str


# ------------------------------------------------
# RAG-lite: load TXT + EPUB from ../docs
# ------------------------------------------------

DOCS_TEXT: str = ""  # big combined context string


def extract_epub_text(path: str) -> str:
    """
    Extract text from an EPUB using ebooklib + BeautifulSoup.
    """
    try:
        book = epub.read_epub(path)
        parts = []
        for item in book.get_items():
            # Only extract XHTML/HTML content
            if item.get_type() == ITEM_DOCUMENT:
                html = item.get_body_content()
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts).strip()
    except Exception as e:
        print(f"[RAG] Error extracting EPUB {path}: {e}")
        return ""


def load_docs_text() -> None:
    """
    Read all .txt and .epub files from ../docs and concatenate them
    into one big context string. This function rescans the folder
    each time it is called, so new files are picked up automatically.
    """
    global DOCS_TEXT

    docs_dir = os.path.join(BASE_DIR, "docs")
    docs_dir = os.path.abspath(docs_dir)

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
            # ignore other file types for now
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


def maybe_add_rag_context(messages: List[dict], use_rag: bool) -> List[dict]:
    """
    If use_rag is True, prepend a system message with DOCS_TEXT.
    """
    if not use_rag:
        return messages

    ensure_docs_text_loaded()

    if not DOCS_TEXT:
        return messages

    system_content = (
        "You are an assistant with access to my personal notes and documents.\n"
        "Use the following context when it is relevant to the user's question. "
        "If it is not relevant, answer normally.\n\n"
        f"CONTEXT:\n{DOCS_TEXT}\n\n"
    )

    return [{"role": "system", "content": system_content}] + messages


# ------------------------------------------------
# Non-streaming chat (for completeness)
# ------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    base_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    messages_for_model = maybe_add_rag_context(base_messages, req.use_rag)

    ollama_payload = {
        "model": req.model,
        "messages": messages_for_model,
        "stream": False,
    }

    resp = requests.post("http://localhost:11434/api/chat", json=ollama_payload)
    resp.raise_for_status()
    data = resp.json()

    answer = data["message"]["content"]
    return ChatResponse(answer=answer, model=req.model)


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

    def generate():
        ollama_payload = {
            "model": req.model,
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
