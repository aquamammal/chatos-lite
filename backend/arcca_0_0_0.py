import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


# ============================================================
# Paths for YOUR layout
# ============================================================

HERE = Path(__file__).resolve()
# arcca_0_0_0.py is: chatos-lite/backend/arcca_0_0_0.py
PROJECT_ROOT = HERE.parents[1]              # chatos-lite/
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
FRONTEND_DIR = PROJECT_ROOT / "static"
MEMORY_DB_PATH = DATA_DIR / "memory.db"

print("[ARCCA] PROJECT_ROOT:", PROJECT_ROOT)
print("[ARCCA] FRONTEND_DIR:", FRONTEND_DIR, "exists:", FRONTEND_DIR.exists())
print("[ARCCA] DOCS_DIR:", DOCS_DIR, "exists:", DOCS_DIR.exists())
print("[ARCCA] MEMORY_DB:", MEMORY_DB_PATH)

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OLLAMA_URL = "http://localhost:11434/api/chat"


# ============================================================
# FastAPI app + static frontend
# ============================================================

app = FastAPI()

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    print("[ARCCA] WARNING: static/ not found – check FRONTEND_DIR")

@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"error": f"index.html not found in {FRONTEND_DIR}"},
            status_code=500,
        )
    return FileResponse(str(index_path))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_loaded": len(DOC_CHUNKS),
        "memory_db": str(MEMORY_DB_PATH),
    }


# ============================================================
# SQLite long-term memory
# ============================================================

def get_db():
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def save_message(session_id: str, role: str, content: str) -> None:
    if not content.strip():
        return
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()

def fetch_recent_memory(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    conn = get_db()
    try:
        cur = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    rows = list(reversed(rows))  # chronological
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def build_memory_chat_history(session_id: str, limit: int = 40) -> List[Dict[str, str]]:
    """
    Convert stored memory rows into actual chat turns so the model
    treats them as real conversation, not just instructions.
    """
    mem = fetch_recent_memory(session_id, limit=limit)
    chat_like: List[Dict[str, str]] = []
    for mm in mem:
        role = "user" if mm["role"] == "user" else "assistant"
        chat_like.append({"role": role, "content": mm["content"]})
    return chat_like

init_db()


# ============================================================
# RAG: load docs from data/docs
# ============================================================

class DocChunk(BaseModel):
    source: str
    text: str

DOC_CHUNKS: List[DocChunk] = []

def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    text = text.replace("\r", " ")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paras:
        if cur_len + len(p) + 2 > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def _load_txt_md(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return _chunk_text(f.read())
    except Exception as e:
        print(f"[RAG] Error TXT/MD {path}: {e}")
        return []

def _load_epub(path: Path) -> List[str]:
    chunks: List[str] = []
    try:
        book = epub.read_epub(str(path))
        texts: List[str] = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                texts.append(soup.get_text(separator="\n"))
        if texts:
            chunks = _chunk_text("\n\n".join(texts))
    except Exception as e:
        print(f"[RAG] Error EPUB {path}: {e}")
    return chunks

def _load_pdf(path: Path) -> List[str]:
    texts: List[str] = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    texts.append(t)
            except Exception:
                continue
    except Exception as e:
        print(f"[RAG] Error PDF {path}: {e}")
        return []
    return _chunk_text("\n\n".join(texts)) if texts else []

def load_all_docs() -> List[DocChunk]:
    chunks: List[DocChunk] = []
    if not DOCS_DIR.exists():
        print("[RAG] DOCS_DIR does not exist, skipping load.")
        return chunks

    print("[RAG] Loading docs from", DOCS_DIR)
    for path in DOCS_DIR.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        file_chunks: List[str] = []
        if ext in {".txt", ".md"}:
            file_chunks = _load_txt_md(path)
        elif ext == ".epub":
            file_chunks = _load_epub(path)
        elif ext == ".pdf":
            file_chunks = _load_pdf(path)
        else:
            continue

        for c in file_chunks:
            chunks.append(DocChunk(source=str(path.relative_to(DOCS_DIR)), text=c))

    print(f"[RAG] Loaded {len(chunks)} chunks.")
    return chunks

DOC_CHUNKS = load_all_docs()

def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in s.split() if len(w) > 2]

def build_rag_context(query: str, max_chunks: int = 5) -> str:
    if not DOC_CHUNKS:
        return ""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return ""

    scored: List[Tuple[int, DocChunk]] = []
    for ch in DOC_CHUNKS:
        score = len(set(q_tokens) & set(_tokenize(ch.text)))
        if score > 0:
            scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [ch for _, ch in scored[:max_chunks]]

    print(f"[RAG] Query {query!r} -> {len(top_chunks)} chunks (of {len(DOC_CHUNKS)})")
    for i, ch in enumerate(top_chunks[:3]):
        prev = ch.text[:150].replace("\n", " ")
        print(f"[RAG] Chunk {i} from {ch.source}: {prev!r}")

    if not top_chunks:
        return ""

    parts: List[str] = []
    for ch in top_chunks:
        parts.append(f"[Source: {ch.source}]\n{ch.text}")
    return "\n\n---\n\n".join(parts)


# ============================================================
# Brave web search
# ============================================================

def brave_web_search(query: str, timeout_s: int = 6) -> Tuple[str, bool]:
    """
    Returns (context, timed_out_flag)
    """
    if not BRAVE_API_KEY:
        print("[BRAVE] No API key; skipping.")
        return "", True

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    params = {"q": query, "count": 5}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout_s)
        if r.status_code != 200:
            print("[BRAVE] Status", r.status_code, "Body:", r.text[:300])
            return "", True
        data = r.json()
    except requests.Timeout:
        print("[BRAVE] Timed out.")
        return "", True
    except Exception as e:
        print("[BRAVE] Error:", e)
        return "", True

    snippets: List[str] = []
    for item in data.get("web", {}).get("results", []):
        title = item.get("title", "")
        desc = item.get("description", "")
        url_item = item.get("url", "")
        snippets.append(
            f"Title: {title}\nURL: {url_item}\nSnippet: {desc}"
        )

    print(f"[BRAVE] Got {len(snippets)} results.")
    return "\n\n---\n\n".join(snippets), False


# ============================================================
# Request model (matches your JS, with optional session_id)
# ============================================================

class ChatReq(BaseModel):
    messages: List[Dict[str, Any]]
    model: str
    use_docs: bool = False
    use_web: bool = False
    session_id: Optional[str] = None


# ============================================================
# Helpers for prompts
# ============================================================

def extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return str(messages[-1].get("content", "")) if messages else ""

def build_system_message(req: ChatReq) -> Tuple[str, Optional[bool]]:
    """
    Returns (system_message, web_timed_out_flag_or_None).
    Memory is handled separately as real chat history.
    """
    parts: List[str] = [
        "You are a helpful assistant running locally on the user's machine."
    ]

    query = extract_last_user_message(req.messages)

    # RAG docs
    if req.use_docs:
        rag_ctx = build_rag_context(query)
        if rag_ctx:
            parts.append(
                "Relevant excerpts from the user's local documents. "
                "Use them when they help answer the question:\n\n" + rag_ctx
            )
        else:
            parts.append("No relevant local document excerpts were found.")

    # Brave web
    web_flag: Optional[bool] = None
    if req.use_web:
        web_ctx, timed_out = brave_web_search(query)
        web_flag = timed_out
        if web_ctx:
            parts.append(
                "Recent web search results that may be helpful:\n\n" + web_ctx
            )
        else:
            if timed_out:
                parts.append("Web search timed out or was unavailable.")
            else:
                parts.append("Web search returned no useful results.")

    return "\n\n---\n\n".join(parts), web_flag


# ============================================================
# Ollama streaming with assistant memory save
# ============================================================

def stream_ollama_with_memory(
    payload: Dict[str, Any],
    session_id: str,
):
    print("[OLLAMA] model:", payload.get("model"))
    acc: List[str] = []
    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=600,
        ) as r:
            if r.status_code != 200:
                body = r.text
                print("---- OLLAMA ERROR ----")
                print("Status:", r.status_code)
                print("Body:", body[:1000])
                print("----------------------")
                err = f"[Backend error: {r.status_code} from Ollama: {body}]"
                yield err
                acc.append(err)
            else:
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message") or {}
                        content = msg.get("content", "")
                        if content:
                            acc.append(content)
                            yield content
                    except json.JSONDecodeError:
                        acc.append(line)
                        yield line
    except requests.RequestException as e:
        err = f"[Backend error: {e}]"
        print(err)
        acc.append(err)
        yield err
    finally:
        full = "".join(acc).strip()
        if full:
            save_message(session_id, "assistant", full)


# ============================================================
# Main /api/chat-stream endpoint
# ============================================================

@app.post("/api/chat-stream")
def api_chat_stream(req: ChatReq):
    # Choose a session id – if frontend doesn’t send, use "default"
    session_id = req.session_id or "default"

    # Store the latest user message in long-term memory
    user_msg = extract_last_user_message(req.messages)
    save_message(session_id, "user", user_msg)

    # Build system message (RAG + web only)
    system_message, web_flag = build_system_message(req)

    # Build message list for Ollama
    messages_for_ollama: List[Dict[str, str]] = []

    # 1. Inject memory as real chat history FIRST
    memory_chat = build_memory_chat_history(session_id, limit=40)
    if memory_chat:
        messages_for_ollama.extend(memory_chat)

    # 2. Then system message
    messages_for_ollama.append({"role": "system", "content": system_message})

    # 3. Then the current session messages from frontend
    for m in req.messages:
        messages_for_ollama.append(
            {
                "role": m.get("role", "user"),
                "content": str(m.get("content", "")),
            }
        )

    payload = {
        "model": req.model,
        "messages": messages_for_ollama,
        "stream": True,
    }

    stream = stream_ollama_with_memory(payload, session_id)
    resp = StreamingResponse(stream, media_type="text/plain")

    if req.use_web:
        if web_flag is True:
            resp.headers["X-Web-Timeout"] = "1"
        elif web_flag is False:
            resp.headers["X-Web-Timeout"] = "0"

    return resp
