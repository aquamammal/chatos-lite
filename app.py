import os
import json
from typing import List

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Serve static frontend from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


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


# -------------------------
# RAG-lite: just load docs text, no embeddings
# -------------------------

DOCS_TEXT: str = ""
DOCS_LOADED: bool = False


def load_docs_text():
    """
    Read all .txt files from ./docs and concatenate them into one big context string.
    No embeddings, no similarity search – just raw text context.
    """
    global DOCS_TEXT, DOCS_LOADED
    if DOCS_LOADED:
        return

    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.isdir(docs_dir):
        print("[RAG] No docs directory found, skipping.")
        DOCS_LOADED = True
        return

    parts = []
    print(f"[RAG] Loading docs text from {docs_dir} ...")
    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".txt"):
            continue
        full_path = os.path.join(docs_dir, fname)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"[RAG] Error reading {full_path}: {e}")
            continue

        if not text:
            continue

        parts.append(f"[{fname}]\n{text}")
        print(f"[RAG] Loaded {fname}")

    DOCS_TEXT = "\n\n".join(parts)
    DOCS_LOADED = True
    print(f"[RAG] Total docs size: {len(DOCS_TEXT)} chars")


def ensure_docs_text_loaded():
    if not DOCS_LOADED:
        load_docs_text()


def maybe_add_rag_context(messages: List[dict], use_rag: bool) -> List[dict]:
    """
    If use_rag is true, add a system message with all docs text as context.
    """
    if not use_rag:
        return messages

    ensure_docs_text_loaded()
    if not DOCS_TEXT:
        return messages

    system_content = (
        "You are an assistant with access to my personal notes.\n"
        "Use the following context from my documents when it is relevant "
        "to the user's question. If it is not relevant, answer normally.\n\n"
        f"CONTEXT:\n{DOCS_TEXT}\n\n"
    )
    return [{"role": "system", "content": system_content}] + messages


# -------------------------
# Non-streaming chat (kept for completeness)
# -------------------------

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


# -------------------------
# Streaming chat with RAG-lite
# -------------------------

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
