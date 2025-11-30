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
from PyPDF2 import PdfReader  # PDF support

# ------------------------------------------------
# Paths / FastAPI setup
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static_v2")   # 👈 v2 frontend
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
    use_docs: bool = False   # local TXT/EPUB/PDF/notes
    use_web: bool = False    # DuckDuckGo web search


class ChatResponse(BaseModel):
    answer: str
    model: str


# ------------------------------------------------
# Globals
# ------------------------------------------------

FULL_DOC_TEXT: str = ""
NOTES_TEXT: str = ""


def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# ------------------------------------------------
# EPUB / PDF / TXT extraction (NO TRUNCATION)
# ------------------------------------------------

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
    """
    global FULL_DOC_TEXT, NOTES_TEXT

    ALL: List[str] = []
    total_chars = 0

    # Personal notes
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
            total_chars += len(text)
            print(f"[RAG] Loaded {fname} ({len(text)} chars)")

    FULL_DOC_TEXT = "\n\n".join(ALL)
    print(f"[RAG] TOTAL LOCAL TEXT (NO LIMIT) = {len(FULL_DOC_TEXT)} chars")


def ensure_loaded() -> None:
    load_all_text()


# ------------------------------------------------
# Web search (DuckDuckGo)
# ------------------------------------------------

def web_search_duckduckgo(query: str, max_results: int = 3) -> str:
    """
    Simple DuckDuckGo web search.
    Returns a text block with titles, snippets, and URLs.
    Tries a couple of different selector patterns and has a fallback.
    """
    query = (query or "").strip()
    if not query:
        return ""

    print(f"[WEB] DuckDuckGo search for: {query!r}")
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WEB] Search request error: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    # --- Strategy 1: classic DuckDuckGo HTML layout ---
    for res in soup.select("div.result, article[data-result]"):
        title_tag = (
            res.select_one("a.result__a")
            or res.select_one("a[href]")
        )
        snippet_tag = (
            res.select_one(".result__snippet")
            or res.select_one("div.result__snippet")
            or res.select_one("p")
        )

        if not title_tag:
            continue

        href = title_tag.get("href")
        title = title_tag.get_text(" ", strip=True)
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""

        if not title and not snippet:
            continue

        results.append((title, snippet, href))
        if len(results) >= max_results:
            break

    # --- Strategy 2: very loose fallback if Strategy 1 fails ---
    if not results:
        print("[WEB] No parsable structured results, using fallback link scan.")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            title = a.get_text(" ", strip=True)
            if not title:
                continue
            # skip nav/duckduckgo internal links
            if "duckduckgo.com" in href:
                continue
            if href.startswith("#"):
                continue
            results.append((title, "", href))
            if len(results) >= max_results:
                break

    if not results:
        print("[WEB] Still no usable results from DuckDuckGo.")
        return ""

    lines: list[str] = []
    for i, (title, snippet, href) in enumerate(results, start=1):
        lines.append(f"[{i}] {title}")
        if snippet:
            lines.append(snippet)
        if href:
            lines.append(href)
        lines.append("")

    block = "\n".join(lines).strip()
    print(f"[WEB] Returning {len(results)} web results.")
    return block



# ------------------------------------------------
# Build hybrid system prompt
# ------------------------------------------------

def build_hybrid_system_message(
    last_user_question: str,
    use_docs: bool,
    enable_web: bool,
) -> Optional[str]:
    ensure_loaded()

    global FULL_DOC_TEXT, NOTES_TEXT

    if not NOTES_TEXT and not (use_docs and FULL_DOC_TEXT) and not enable_web:
        return None

    parts: List[str] = [
        "You are a hybrid assistant with access to:",
        "1) PERSONAL NOTES (about me)",
        "2) FULL LOCAL DOCUMENTS (TXT, EPUB, PDF) (if included)",
        "3) LIVE WEB SEARCH RESULTS (DuckDuckGo) (if included)",
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
        web_block = web_search_duckduckgo(last_user_question, max_results=3)
        parts.append("WEB SEARCH RESULTS (DuckDuckGo):")
        if web_block:
            parts.append(web_block)
        else:
            parts.append("[No results or search failed]")
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
    We call Ollama non-streaming and stream the final answer as a single chunk.
    """
    base_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    msgs = maybe_add_hybrid_context(base_msgs, req.use_docs, req.use_web)

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

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_hybrid_two_toggles:app", host="127.0.0.1", port=8001, reload=True)
