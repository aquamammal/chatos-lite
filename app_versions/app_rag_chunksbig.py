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
# Request / response models (match your frontend)
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
# RAG: notes + chunked docs (MAX MODE)
# ------------------------------------------------

class DocChunk:
    """
    Simple structure for a chunk of document text.
    """
    __slots__ = ("source", "text", "text_lower")

    def __init__(self, source: str, text: str):
        self.source = source
        self.text = text
        self.text_lower = text.lower()


DOC_CHUNKS: List[DocChunk] = []
NOTES_TEXT: str = ""


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


def chunk_text(text: str, chunk_size: int = 10000, overlap: int = 600) -> List[str]:
    """
    MAX mode:
    - Large chunks (~3500 chars)
    - Generous overlap (~600 chars)
    Good for novels / long docs.
    """
    chunks: List[str] = []
    if not text:
        return chunks

    start = 0
    length = len(text)
    step = max(1, chunk_size - overlap)

    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def load_personal_notes() -> str:
    """
    Read docs/notes.txt if it exists.
    """
    notes_path = os.path.join(DOCS_DIR, "notes.txt")
    try:
        if os.path.isfile(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                notes = f.read().strip()
                print(f"[RAG] Loaded personal notes, length={len(notes)} chars")
                return notes
    except Exception as e:
        print(f"[RAG] Error reading personal notes: {e}")
    return ""


def load_docs_and_chunks() -> None:
    """
    Load all .txt and .epub files in ../docs, create overlapping chunks,
    and store in DOC_CHUNKS. Also loads NOTES_TEXT.
    """
    global DOC_CHUNKS, NOTES_TEXT

    DOC_CHUNKS = []
    NOTES_TEXT = load_personal_notes()

    docs_dir = os.path.abspath(DOCS_DIR)
    if not os.path.isdir(docs_dir):
        print(f"[RAG] No docs directory at {docs_dir}, skipping.")
        return

    print(f"[RAG] Scanning docs in {docs_dir} ...")
    total_chars = 0
    total_chunks = 0

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

        total_chars += len(text)
        file_chunks = chunk_text(text)
        n_chunks = len(file_chunks)
        total_chunks += n_chunks

        for chunk in file_chunks:
            DOC_CHUNKS.append(DocChunk(source=fname, text=chunk))

        print(f"[RAG] Loaded {fname} ({len(text)} chars, {n_chunks} chunks)")

    print(f"[RAG] Total docs chars: {total_chars}")
    print(f"[RAG] Total chunks: {total_chunks}")


def ensure_docs_loaded() -> None:
    """
    Rescan docs every time so new files are picked up automatically.
    """
    load_docs_and_chunks()


def score_chunk(query: str, chunk: DocChunk) -> int:
    """
    Very simple scoring: count of query tokens that appear in the chunk.
    """
    q = query.lower().strip()
    if not q:
        return 0

    tokens = {t for t in q.replace("\n", " ").split(" ") if t}
    if not tokens:
        return 0

    text = chunk.text_lower
    score = 0
    for t in tokens:
        if t in text:
            score += 1
    return score


def build_rag_system_message(
    messages: List[Dict[str, Any]],
    max_context_chars: int = 1700000,
    max_chunks: int = 150,
) -> Optional[str]:
    """
    MAX mode:
    - Up to ~48k characters of RAG context
    - Up to 16 best-matching chunks
    Tuned for Qwen / Mixtral / Llama on a 4090.
    """
    ensure_docs_loaded()

    # Find last user message
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    if not last_user and not NOTES_TEXT and not DOC_CHUNKS:
        return None

    scored: List[tuple[int, DocChunk]] = []
    for chunk in DOC_CHUNKS:
        s = score_chunk(last_user, chunk)
        if s > 0:
            scored.append((s, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected_chunks: List[DocChunk] = []
    current_chars = 0

    # Oversample aggressively, pick best until we hit limits
    for _, chunk in scored[:max_chunks * 10]:
        if len(selected_chunks) >= max_chunks:
            break
        chunk_len = len(chunk.text)
        if current_chars + chunk_len > max_context_chars:
            continue
        selected_chunks.append(chunk)
        current_chars += chunk_len

    if not selected_chunks and not NOTES_TEXT:
        return None

    parts: List[str] = [
        "You are an assistant with access to my personal notes and relevant document excerpts.",
        "Use PERSONAL NOTES first for anything about me.",
        "Use DOCUMENT EXCERPTS when they help answer the user's question.",
        "",
    ]

    if NOTES_TEXT:
        parts.append("PERSONAL NOTES:")
        parts.append(NOTES_TEXT)
        parts.append("")

    if selected_chunks:
        parts.append("DOCUMENT EXCERPTS (max mode):")
        for i, chunk in enumerate(selected_chunks, start=1):
            header = f"[{chunk.source} | chunk #{i}]"
            parts.append(header)
            parts.append(chunk.text)
            parts.append("")

    system_content = "\n".join(parts)
    print(
        f"[RAG] MAX MODE: Selected {len(selected_chunks)} chunks, "
        f"total={len(system_content)} chars"
    )
    return system_content


def maybe_add_rag_context(
    messages: List[Dict[str, Any]],
    use_rag: bool,
) -> List[Dict[str, Any]]:
    """
    If use_rag is True, prepend a system message with notes + relevant chunks.
    """
    if not use_rag:
        return messages

    system_content = build_rag_system_message(messages)
    if not system_content:
        return messages

    system_msg = {"role": "system", "content": system_content}
    return [system_msg] + messages


def get_model_name(req: ChatRequest) -> str:
    """
    Determine which model to use.
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
# Streaming chat (works with your current frontend)
# ------------------------------------------------

@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):
    """
    Streaming endpoint: optionally prepends notes + relevant chunks as a system message,
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


# ------------------------------------------------
# Run directly (optional)
# ------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_rag_chunks:app", host="127.0.0.1", port=8000, reload=True)
