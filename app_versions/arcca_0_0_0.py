import os
import time
import re
import sqlite3
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


# ================================================================
# Paths / FastAPI setup
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static_v3")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DATA_DIR = os.path.join(BASE_DIR, "Data")  # your directory name exactly

os.makedirs(DATA_DIR, exist_ok=True)

MEM_DB_PATH = os.path.join(DATA_DIR, "memory.db")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ================================================================
# Brave API config
# ================================================================

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

WEB_CACHE: Dict[str, str] = {}
WEB_TIMEOUT_LAST_CALL: Optional[bool] = None


# ================================================================
# SQLite long-term memory (Data/memory.db)
# ================================================================

def init_memory_db() -> None:
    """Create SQLite DB and memory table if missing."""
    conn = sqlite3.connect(MEM_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            topic    TEXT,
            content  TEXT,
            created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_memory(topic: str, content: str) -> None:
    """Store memory row in SQLite."""
    content = (content or "").strip()
    if not content:
        return
    conn = sqlite3.connect(MEM_DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memory (topic, content) VALUES (?, ?)",
        (topic[:128], content),
    )
    conn.commit()
    conn.close()


def load_all_memories(max_chars: int = 6000) -> str:
    """
    Return memory dump, limited to max_chars for prompt stability.
    Newest first.
    """
    conn = sqlite3.connect(MEM_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT topic, content FROM memory ORDER BY created DESC, id DESC")
    rows = c.fetchall()
    conn.close()

    parts = []
    total = 0

    for topic, content in rows:
        entry = f"[{topic}] {content}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n".join(parts)


init_memory_db()


# ================================================================
# Chat models and request wrappers
# ================================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    use_docs: bool = False
    use_web: bool = False


class ChatResponse(BaseModel):
    answer: str
    model: str


def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# ================================================================
# Auto-Memory Extraction (simple heuristic rules)
# ================================================================

def auto_update_memory(last_user_text: str, assistant_text: str) -> None:
    """Extract obvious facts and store them as memory."""

    if not last_user_text:
        return

    u_low = last_user_text.lower()

    # Explicit "remember" command
    m = re.search(r"(remember this|remember that|please remember)\s*(.*)", u_low)
    if m:
        save_memory("explicit_remember", last_user_text)
        return

    # Name
    m = re.search(r"my name is\s+(.+)", u_low)
    if m:
        name = m.group(1).strip(" .!?,")
        save_memory("user_name", f"The user's name is {name}.")
        return

    # Location
    m = re.search(r"\bi (live|am living) in\s+(.+)", u_low)
    if m:
        place = m.group(2).strip(" .!?,")
        save_memory("user_location", f"The user lives in {place}.")
        return

    # From
    m = re.search(r"\bi am from\s+(.+)", u_low) or re.search(r"\bi'm from\s+(.+)", u_low)
    if m:
        place = m.group(1).strip(" .!?,")
        save_memory("user_origin", f"The user is from {place}.")
        return

    # Preferences
    pref_patterns = [
        r"my favorite (color|food|movie|song|band|book|place) is\s+(.+)",
        r"i really love\s+(.+)",
        r"i like to\s+(.+)",
    ]
    for pat in pref_patterns:
        m = re.search(pat, u_low)
        if m:
            save_memory("user_preference", last_user_text)
            return

    # Pets
    pet_triggers = ["my dog", "my cat", "i have a dog", "i have a cat", "my puppy", "my kitten"]
    if any(kw in u_low for kw in pet_triggers):
        save_memory("user_pets", last_user_text)
        return

    # Job
    if "i work as " in u_low or "i work at " in u_low:
        save_memory("user_job", last_user_text)
        return
    if ("i am a " in u_low or "i'm a " in u_low) and "fan of" not in u_low and "kind of" not in u_low:
        save_memory("user_job_or_role", last_user_text)
        return

    # Projects
    project_keywords = ["my project is", "i am working on", "i'm working on"]
    if any(kw in u_low for kw in project_keywords):
        save_memory("user_project", last_user_text)
        return

    # Dates
    date_keywords = ["my birthday", "i was born", "anniversary"]
    if any(kw in u_low for kw in date_keywords):
        save_memory("user_dates", last_user_text)
        return


# ================================================================
# Local Docs Extraction (no truncation)
# ================================================================

FULL_DOC_TEXT = ""
DOCS_LOADED = False


def extract_epub_text(path: str) -> str:
    try:
        book = epub.read_epub(path)
    except Exception:
        return ""
    parts = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            parts.append(soup.get_text("\n"))
    return "\n\n".join(parts)


def extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
    except Exception:
        return ""
    return "\n\n".join([(page.extract_text() or "") for page in reader.pages])


def load_all_docs() -> None:
    global FULL_DOC_TEXT, DOCS_LOADED

    if DOCS_LOADED:
        return

    texts = []
    if os.path.isdir(DOCS_DIR):
        for fname in os.listdir(DOCS_DIR):
            path = os.path.join(DOCS_DIR, fname)
            if not os.path.isfile(path):
                continue

            f_low = fname.lower()

            if f_low.endswith(".txt"):
                try:
                    texts.append(f"[{fname}]\n" + open(path, "r", encoding="utf-8").read())
                except:
                    continue
            elif f_low.endswith(".epub"):
                texts.append(f"[{fname}]\n" + extract_epub_text(path))
            elif f_low.endswith(".pdf"):
                texts.append(f"[{fname}]\n" + extract_pdf_text(path))

    FULL_DOC_TEXT = "\n\n".join(texts)
    DOCS_LOADED = True


# ================================================================
# Brave Web Search
# ================================================================

def normalize_query(q: str) -> str:
    return " ".join((q or "").lower().split())


def rewrite_query_for_web(question: str) -> str:
    q = question.strip()
    l = q.lower()

    if "dc" in l and "washington" not in l:
        q = q.replace("dc", "washington dc")

    if any(t in l for t in ["today", "yesterday", "this week", "happened"]) and "news" not in l:
        q = "current news " + q

    return q


def web_search_brave(question: str, max_results: int = 3) -> str:
    global WEB_TIMEOUT_LAST_CALL

    WEB_TIMEOUT_LAST_CALL = None

    if not BRAVE_API_KEY:
        print("[WEB] No API key → skipping.")
        return ""

    rewritten = rewrite_query_for_web(question)
    key = normalize_query(rewritten)

    if key in WEB_CACHE:
        WEB_TIMEOUT_LAST_CALL = False
        return WEB_CACHE[key]

    params = {"q": rewritten, "count": max_results}
    headers = {"X-Subscription-Token": BRAVE_API_KEY}

    attempts = 3
    timeout_flag = False

    for i in range(attempts):
        try:
            r = requests.get(BRAVE_ENDPOINT, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                time.sleep(2)
                continue

            r.raise_for_status()
            data = r.json()
            results = data.get("web", {}).get("results", [])

            if not results:
                WEB_TIMEOUT_LAST_CALL = False
                return ""

            lines = []
            for idx, res in enumerate(results[:max_results]):
                lines.append(f"[{idx+1}] {res.get('title','')}")
                lines.append(res.get("description",""))
                lines.append(res.get("url",""))
                lines.append("")

            block = "\n".join(lines).strip()
            WEB_CACHE[key] = block
            WEB_TIMEOUT_LAST_CALL = False
            return block

        except requests.exceptions.Timeout:
            timeout_flag = True
            time.sleep(2)
        except:
            return ""

    WEB_TIMEOUT_LAST_CALL = True if timeout_flag else None
    return ""


# ================================================================
# Build hybrid RAG context (memory + docs + web)
# ================================================================

def build_hybrid_system_message(question: str, use_docs: bool, use_web: bool) -> str:
    parts = [
        "You are ARCCA, a hybrid local assistant with:",
        "- LONG-TERM MEMORY (SQLite)",
        "- LOCAL DOCUMENTS (txt/epub/pdf)",
        "- OPTIONAL WEB SEARCH (Brave)",
        "",
        "Always use this priority:",
        "1) Long-term memory",
        "2) Local docs",
        "3) Web search",
        ""
    ]

    mem_dump = load_all_memories()
    if mem_dump:
        parts.append("MEMORIES:")
        parts.append(mem_dump)
        parts.append("")

    if use_docs:
        load_all_docs()
        if FULL_DOC_TEXT:
            parts.append("LOCAL DOCUMENTS:")
            parts.append(FULL_DOC_TEXT)
            parts.append("")

    if use_web:
        web_block = web_search_brave(question)
        parts.append("WEB RESULTS:")
        parts.append(web_block or "[No results]")
        parts.append("")

    return "\n".join(parts)


def maybe_inject_context(messages: List[Dict[str, Any]], use_docs: bool, use_web: bool):
    if not use_docs and not use_web:
        return messages
    last_q = get_last_user_message(messages)
    context = build_hybrid_system_message(last_q, use_docs, use_web)
    return [{"role": "system", "content": context}] + messages


def get_model_name(req: ChatRequest) -> str:
    return req.model or "qwen2.5:7b"


# ================================================================
# Chat Endpoints
# ================================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    base_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    msgs = maybe_inject_context(base_msgs, req.use_docs, req.use_web)

    model = get_model_name(req)
    payload = {"model": model, "messages": msgs, "stream": False}

    r = requests.post("http://localhost:11434/api/chat", json=payload)
    r.raise_for_status()
    data = r.json()

    answer = data["message"]["content"]

    # Auto-memory
    last_user = get_last_user_message(base_msgs)
    auto_update_memory(last_user, answer)

    return ChatResponse(answer=answer, model=model)


@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):
    base_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    msgs = maybe_inject_context(base_msgs, req.use_docs, req.use_web)

    model = get_model_name(req)
    web_flag = WEB_TIMEOUT_LAST_CALL

    def generate():
        try:
            r = requests.post("http://localhost:11434/api/chat",
                              json={"model": model, "messages": msgs, "stream": False})
            r.raise_for_status()
            text = r.json()["message"]["content"]
        except Exception as e:
            text = f"[Backend error: {e}]"

        # Auto-memory
        last_user = get_last_user_message(base_msgs)
        auto_update_memory(last_user, text)

        yield text

    headers = {}
    if req.use_web:
        if web_flag is True:
            headers["X-Web-Timeout"] = "1"
        elif web_flag is False:
            headers["X-Web-Timeout"] = "0"

    return StreamingResponse(generate(), media_type="text/plain", headers=headers)


# ================================================================
# RUN SERVER (uvicorn)
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_versions.arcca_0_0_0:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
