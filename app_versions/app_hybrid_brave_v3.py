import os
import time
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader  # PDF support


# ------------------------------------------------
# Paths / FastAPI setup
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static_v3")   # v3 frontend (docs + web)
DOCS_DIR = os.path.join(BASE_DIR, "docs")

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ------------------------------------------------
# Brave API config
# ------------------------------------------------

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")  # set in ~/.bashrc
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# In-memory cache: normalized_query -> results_block
WEB_CACHE: Dict[str, str] = {}

# Web timeout flag for last Brave call:
# - True  -> Brave timed out after retries
# - False -> Brave used successfully (API or cache) / no timeout
# - None  -> Brave not used / not applicable / API key missing
WEB_TIMEOUT_LAST_CALL: Optional[bool] = None


# ------------------------------------------------
# Request / response models
# ------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    use_docs: bool = False   # local TXT/EPUB/PDF/notes
    use_web: bool = False    # Brave web search


class ChatResponse(BaseModel):
    answer: str
    model: str


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def normalize_query(q: str) -> str:
    return " ".join((q or "").lower().split())


def rewrite_query_for_web(question: str) -> str:
    """
    Very simple query rewriter to make vague questions more search-friendly.
    Examples:
      "What happened in DC this week?" -> "current news Washington DC this week"
    """
    q = (question or "").strip()
    if not q:
        return q

    lowered = q.lower()

    # Expand "DC" -> "Washington DC" if not already present
    if "dc" in lowered and "washington" not in lowered:
        q = q.replace("DC", "Washington DC").replace("dc", "Washington DC")

    # Add "news" for time-based queries
    time_words = ["today", "yesterday", "this week", "this weekend", "tonight"]
    if any(tw in lowered for tw in time_words) and "news" not in lowered:
        q = "current news " + q

    # Fallback: if the question looks like a question about events, prepend "news"
    if "happened" in lowered and "news" not in q.lower():
        q = "news " + q

    return q


# ------------------------------------------------
# EPUB / PDF / TXT extraction (NO TRUNCATION)
# ------------------------------------------------

FULL_DOC_TEXT: str = ""
NOTES_TEXT: str = ""
DOCS_LOADED: bool = False  # only load docs once, and only when use_docs is True


def extract_epub_text(path: str) -> str:
    print(f"[RAG] Extracting EPUB: {path}")
    try:
        book = epub.read_epub(path)
    except Exception as e:
        print(f"[RAG] EPUB read error: {e}")
        return ""

    parts: List[str] = []
    for item in book.get_items():
        try:
            if item.get_type() == ITEM_DOCUMENT:
                html = item.get_body_content()
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    parts.append(text)
        except Exception as e:
            print(f"[RAG] EPUB item error: {e}")
            continue

    combined = "\n\n".join(parts).strip()
    print(f"[RAG] EPUB extracted {len(combined)} chars")
    return combined


def extract_pdf_text(path: str) -> str:
    print(f"[RAG] Extracting PDF: {path}")
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"[RAG] PDF read error: {e}")
        return ""

    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        except Exception as e:
            print(f"[RAG] PDF page {i} error: {e}")
            continue

    combined = "\n\n".join(parts).strip()
    print(f"[RAG] PDF extracted {len(combined)} chars")
    return combined


def load_all_text() -> None:
    """
    Load ALL .txt, .epub, and .pdf files in DOCS_DIR.
    NO LIMITS. NO TRUNCATION.
    Only called when RAG is actually enabled.
    """
    global FULL_DOC_TEXT, NOTES_TEXT, DOCS_LOADED

    ALL: List[str] = []

    NOTES_TEXT = ""
    notes_path = os.path.join(DOCS_DIR, "notes.txt")
    if os.path.isfile(notes_path):
        try:
            with open(notes_path, "r", encoding="utf-8") as f:
                NOTES_TEXT = f.read().strip()
                print(f"[RAG] Loaded notes.txt ({len(NOTES_TEXT)} chars)")
        except Exception as e:
            print(f"[RAG] notes.txt read error: {e}")

    if not os.path.isdir(DOCS_DIR):
        print(f"[RAG] docs directory not found at {DOCS_DIR}")
        FULL_DOC_TEXT = ""
        DOCS_LOADED = True
        return

    print(f"[RAG] Scanning docs in {DOCS_DIR} ...")

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
            except Exception as e:
                print(f"[RAG] TXT read error for {full_path}: {e}")
                continue

        elif lower.endswith(".epub"):
            text = extract_epub_text(full_path)

        elif lower.endswith(".pdf"):
            text = extract_pdf_text(full_path)

        else:
            continue

        if text.strip():
            ALL.append(f"[{fname}]\n{text}")
            print(f"[RAG] Loaded {fname} ({len(text)} chars)")

    FULL_DOC_TEXT = "\n\n".join(ALL)
    DOCS_LOADED = True
    print(f"[RAG] TOTAL LOCAL TEXT (NO LIMIT) = {len(FULL_DOC_TEXT)} chars")


def ensure_loaded(use_docs: bool) -> None:
    """
    Only load the docs when RAG is actually enabled.
    If use_docs is False, we do nothing.
    """
    global DOCS_LOADED
    if not use_docs:
        return
    if DOCS_LOADED:
        return
    load_all_text()


# ------------------------------------------------
# Web search via Brave API (with retries, caching, logging)
# ------------------------------------------------

def web_search_brave(question: str, max_results: int = 3) -> str:
    """
    Use Brave Search API with:
      - query rewriting
      - in-memory caching
      - retries with backoff
      - detailed logging (including exact result block)
      - sets WEB_TIMEOUT_LAST_CALL for frontend timeout indicator
    """
    global WEB_TIMEOUT_LAST_CALL

    # Default: assume no timeout, will refine below
    WEB_TIMEOUT_LAST_CALL = None

    if not BRAVE_API_KEY:
        print("[WEB] BRAVE_API_KEY not set. Skipping web search.")
        WEB_TIMEOUT_LAST_CALL = None
        return ""

    original_query = (question or "").strip()
    rewritten = rewrite_query_for_web(original_query)
    print(f"[WEB] Original query: {original_query!r}")
    print(f"[WEB] Rewritten query for Brave: {rewritten!r}")

    norm_key = normalize_query(rewritten)
    if norm_key in WEB_CACHE:
        print("[WEB] Cache hit for query.")
        block = WEB_CACHE[norm_key]
        print("[WEB] Cached Brave results block:\n" + block + "\n" + "-" * 60)
        WEB_TIMEOUT_LAST_CALL = False  # cache hit => no timeout
        return block

    params = {
        "q": rewritten,
        "count": max_results,
        "source": "web",
        "country": "us",
        "safesearch": "moderate",
    }
    headers = {
        "X-Subscription-Token": BRAVE_API_KEY,
        "Accept": "application/json",
    }

    attempts = 3
    last_error = None
    timeout_happened = False

    for attempt in range(1, attempts + 1):
        try:
            print(f"[WEB] Brave request attempt {attempt}/{attempts} ...")
            resp = requests.get(
                BRAVE_ENDPOINT,
                headers=headers,
                params=params,
                timeout=10,
            )
            if resp.status_code == 429:
                last_error = f"HTTP 429 (rate limited), attempt {attempt}"
                print(f"[WEB] {last_error}")
                time.sleep(2 * attempt)
                continue
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}, attempt {attempt}"
                print(f"[WEB] {last_error}")
                time.sleep(2 * attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                print("[WEB] Brave returned no web results.")
                WEB_TIMEOUT_LAST_CALL = False
                return ""

            lines: List[str] = []
            used = 0
            for res in web_results:
                title = res.get("title") or ""
                desc = res.get("description") or res.get("snippet") or ""
                url = res.get("url") or ""

                if not title and not desc:
                    continue

                used += 1
                lines.append(f"[{used}] {title}")
                if desc:
                    lines.append(desc)
                if url:
                    lines.append(url)
                lines.append("")

                if used >= max_results:
                    break

            if not lines:
                print("[WEB] Brave had results but none usable.")
                WEB_TIMEOUT_LAST_CALL = False
                return ""

            block = "\n".join(lines).strip()
            print(f"[WEB] Returning {used} Brave results.")
            print("[WEB] Brave results block:\n" + block + "\n" + "-" * 60)

            WEB_CACHE[norm_key] = block
            WEB_TIMEOUT_LAST_CALL = False
            return block

        except requests.exceptions.Timeout:
            timeout_happened = True
            last_error = f"Timeout on attempt {attempt}"
            print(f"[WEB] {last_error}")
            time.sleep(2 * attempt)
        except Exception as e:
            last_error = f"Exception on attempt {attempt}: {e}"
            print(f"[WEB] {last_error}")
            time.sleep(2 * attempt)

    # All attempts failed
    if timeout_happened:
        print("[WEB] Web search timeout indicator: Brave timed out after retries.")
        WEB_TIMEOUT_LAST_CALL = True
    else:
        WEB_TIMEOUT_LAST_CALL = None

    if last_error:
        print(f"[WEB] Last Brave error: {last_error}")

    return ""


# ------------------------------------------------
# Build hybrid system prompt
# ------------------------------------------------

def build_hybrid_system_message(
    last_user_question: str,
    use_docs: bool,
    enable_web: bool,
) -> Optional[str]:
    # Only loads docs if use_docs is True
    ensure_loaded(use_docs)

    global FULL_DOC_TEXT, NOTES_TEXT

    if not NOTES_TEXT and not (use_docs and FULL_DOC_TEXT) and not enable_web:
        return None

    parts: List[str] = [
        "You are a hybrid assistant with access to:",
        "1) PERSONAL NOTES about me.",
        "2) FULL LOCAL DOCUMENTS (TXT, EPUB, PDF) from my docs folder (if included).",
        "3) LIVE WEB SEARCH RESULTS from Brave Search (if included).",
        "",
        "Use this priority:",
        "- PERSONAL NOTES first for anything about me.",
        "- Then LOCAL DOCUMENTS when relevant.",
        "- Only when those are insufficient or outdated, consult WEB SEARCH RESULTS.",
        "",
    ]

    if NOTES_TEXT:
        parts.append("PERSONAL NOTES:")
        parts.append(NOTES_TEXT)
        parts.append("")

    if use_docs and FULL_DOC_TEXT:
        parts.append("FULL LOCAL DOCUMENT DUMP:")
        parts.append(FULL_DOC_TEXT)
        parts.append("")

    if enable_web:
        web_block = web_search_brave(last_user_question, max_results=3)
        parts.append("WEB SEARCH RESULTS (Brave):")
        if web_block:
            parts.append(web_block)
        else:
            parts.append("[No web results available (timeout / error / none found)]")
        parts.append("")

    system_content = "\n".join(parts)
    print(f"[HYBRID] System message size = {len(system_content)} chars")
    return system_content


def maybe_add_hybrid_context(
    messages: List[Dict[str, Any]],
    use_docs: bool,
    use_web: bool,
) -> List[Dict[str, Any]]:
    if not use_docs and not use_web:
        return messages

    last_user = get_last_user_message(messages)
    system_content = build_hybrid_system_message(
        last_user_question=last_user,
        use_docs=use_docs,
        enable_web=use_web,
    )

    if not system_content:
        return messages

    system_msg = {"role": "system", "content": system_content}
    return [system_msg] + messages


def get_model_name(req: ChatRequest) -> str:
    return req.model or "qwen2.5:7b"


# ------------------------------------------------
# Chat endpoints
# ------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    base_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    msgs = maybe_add_hybrid_context(base_msgs, req.use_docs, req.use_web)

    model = get_model_name(req)
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
    }

    r = requests.post("http://localhost:11434/api/chat", json=payload)
    r.raise_for_status()
    data = r.json()

    answer = data["message"]["content"]
    return ChatResponse(answer=answer, model=model)


@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):
    """
    Call Ollama non-streaming and stream the final answer as a single chunk.
    Also exposes a web-search timeout indicator via the X-Web-Timeout header:
      "1" -> Brave timed out after retries
      "0" -> Brave used successfully (or cache hit / no timeout)
      (absent) -> web search not used or not applicable
    """
    global WEB_TIMEOUT_LAST_CALL

    base_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    msgs = maybe_add_hybrid_context(base_msgs, req.use_docs, req.use_web)

    # Snapshot the timeout flag right after building context
    # (this is when Brave is called, if enabled)
    web_timeout_flag = WEB_TIMEOUT_LAST_CALL

    model = get_model_name(req)

    def generate():
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
        }

        try:
            r = requests.post("http://localhost:11434/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "") or ""
        except Exception as e:
            text = f"[Backend error calling model: {e}]"

        yield text

    # Build headers for timeout indicator
    headers: Dict[str, str] = {}
    if req.use_web:
        if web_timeout_flag is True:
            headers["X-Web-Timeout"] = "1"
        elif web_timeout_flag is False:
            headers["X-Web-Timeout"] = "0"
        # if None, don't set header (unknown / not applicable)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_hybrid_brave_v3:app", host="127.0.0.1", port=8001, reload=True)

