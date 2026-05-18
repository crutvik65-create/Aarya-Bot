"""
PropVoice — Real Estate RAG API  v2.4.0
=======================================
Run locally:
    pip install -r requirements.txt
    python app.py  →  http://localhost:8000

Deploy on Render (no Docker):
    buildCommand : pip install -r requirements.txt
    startCommand : gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:$PORT --timeout 120

Folder layout (everything in one repo):
    propvoice/
    ├── app.py
    ├── requirements.txt
    ├── render.yaml
    ├── .gitignore
    ├── static/          ← frontend files (index.html, css, js …)
    └── RAG/             ← property PDFs / DOCX files
"""

import asyncio
import base64
import hashlib
import logging
import os
import re
import struct
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional
from xml.etree import ElementTree as ET

# ── Silence ChromaDB telemetry completely ─────────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import chromadb


class _ChromaTelemetryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("telemetry" in msg.lower() or "posthog" in msg.lower())


_telem_filter = _ChromaTelemetryFilter()
for _h in logging.root.handlers:
    _h.addFilter(_telem_filter)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").addFilter(_telem_filter)

import httpx
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pypdf import PdfReader

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("propvoice")


class Settings(BaseSettings):
    openai_api_key: str       = Field(...,           env="OPENAI_API_KEY")
    sarvam_api_key: str       = Field(...,           env="SARVAM_API_KEY")
    chroma_persist_dir: str   = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    rag_dir: str              = Field("./RAG",       env="RAG_DIR")
    host: str                 = Field("0.0.0.0",     env="HOST")
    port: int                 = Field(8000,          env="PORT")
    workers: int              = Field(1,             env="WORKERS")

    sarvam_tts_model: str     = Field("bulbul:v3",   env="SARVAM_TTS_MODEL")
    rag_n_results: int        = Field(3,             env="RAG_N_RESULTS")
    embedding_cache_size: int = Field(512,           env="EMBEDDING_CACHE_SIZE")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


cfg = Settings()

_LANG_SPEAKER: dict = {
    "english":  os.getenv("SARVAM_SPEAKER_ENGLISH",  "anushka"),
    "hindi":    os.getenv("SARVAM_SPEAKER_HINDI",    "priya"),
    "gujarati": os.getenv("SARVAM_SPEAKER_GUJARATI", "ritu"),
    "telugu":   os.getenv("SARVAM_SPEAKER_TELUGU",   "ritu"),
    "tamil":    os.getenv("SARVAM_SPEAKER_TAMIL",    "ishita"),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ROBUST DOCX TEXT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

_NS = {
    "w":   "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "mc":  "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "v":   "urn:schemas-microsoft-com:vml",
}
for _prefix, _uri in _NS.items():
    ET.register_namespace(_prefix, _uri)


def _iter_text_nodes(element) -> List[str]:
    texts = []
    for node in element.iter():
        tag = node.tag
        if tag.endswith("}t") and "wordprocessingml" in tag:
            val = (node.text or "").strip()
            if val:
                texts.append(val)
        elif tag.endswith("}t") and "drawingml" in tag:
            val = (node.text or "").strip()
            if val:
                texts.append(val)
    return texts


def _extract_xml_part(zf: zipfile.ZipFile, part_path: str) -> str:
    try:
        with zf.open(part_path) as f:
            tree = ET.parse(f)
        return " ".join(_iter_text_nodes(tree.getroot()))
    except KeyError:
        return ""
    except ET.ParseError as e:
        log.warning("XML parse error in %s: %s", part_path, e)
        return ""


def _list_xml_parts(zf: zipfile.ZipFile, prefix: str) -> List[str]:
    return [name for name in zf.namelist() if name.startswith(prefix)]


def extract_docx_text(file_path: Path) -> str:
    sections: List[str] = []
    with zipfile.ZipFile(str(file_path), "r") as zf:
        for part in ["word/document.xml"]:
            t = _extract_xml_part(zf, part)
            if t:
                sections.append(t)
        for part in _list_xml_parts(zf, "word/header"):
            t = _extract_xml_part(zf, part)
            if t:
                sections.append(t)
        for part in _list_xml_parts(zf, "word/footer"):
            t = _extract_xml_part(zf, part)
            if t:
                sections.append(t)
        for part_name in ["word/footnotes.xml", "word/endnotes.xml", "word/comments.xml"]:
            t = _extract_xml_part(zf, part_name)
            if t:
                sections.append(t)
        for part in _list_xml_parts(zf, "word/charts/"):
            t = _extract_xml_part(zf, part)
            if t:
                sections.append(t)
    return "\n\n".join(s for s in sections if s.strip())


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CHROMADB + RAG SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

_chroma = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
_embed_fn = OpenAIEmbeddingFunction(
    api_key=cfg.openai_api_key,
    model_name="text-embedding-3-small",
)

_retrieval_cache: dict = {}
_CACHE_MAX = cfg.embedding_cache_size


def _cache_key(collection_id: str, query: str) -> str:
    return f"{collection_id}:{hashlib.sha256(query.encode()).hexdigest()}"


def _col_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) < 3:
        name = name + "_col"
    return name[:63]


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".docx", ".doc"):
        try:
            return extract_docx_text(file_path)
        except zipfile.BadZipFile:
            log.warning("%s is not a valid ZIP/DOCX — attempting raw byte read", file_path.name)
            try:
                raw = file_path.read_bytes()
                raw_chunks = re.findall(rb"[\x20-\x7E]{4,}", raw)
                return " ".join(c.decode("ascii", errors="ignore") for c in raw_chunks)
            except Exception as e:
                log.error("Raw read failed for %s: %s", file_path.name, e)
                return ""
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


_SUPPORTED_EXTS = {".pdf", ".docx", ".doc"}
_TEMP_PREFIXES  = ("~$", "~")


def _is_valid_doc(p: Path) -> bool:
    if p.suffix.lower() not in _SUPPORTED_EXTS:
        return False
    if p.name.startswith(_TEMP_PREFIXES):
        return False
    return True


def rag_list_docs() -> List[dict]:
    rag_dir = Path(cfg.rag_dir)
    if not rag_dir.exists():
        return []
    files = sorted([p for p in rag_dir.glob("*") if _is_valid_doc(p)])
    return [
        {
            "id": _col_name(p.name),
            "filename": p.name,
            "display_name": p.stem.replace("_", " ").replace("-", " ").title(),
        }
        for p in files
    ]


def rag_ingest_doc(filename: str) -> dict:
    doc_path = Path(cfg.rag_dir) / filename
    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")
    col_name = _col_name(filename)
    current_hash = _file_md5(str(doc_path))
    try:
        col = _chroma.get_collection(name=col_name, embedding_function=_embed_fn)
        if col.metadata.get("file_hash") == current_hash and col.count() > 0:
            return {"status": "skipped", "collection": col_name, "chunks": col.count()}
        _chroma.delete_collection(col_name)
    except Exception:
        pass
    full_text = _extract_text(doc_path)
    if not full_text.strip():
        log.warning("No text extracted from %s", filename)
        col = _chroma.get_or_create_collection(
            name=col_name, embedding_function=_embed_fn,
            metadata={"file": filename, "file_hash": current_hash},
        )
        return {"status": "empty", "collection": col_name, "chunks": 0}
    chunks = _chunk_text(full_text)
    col = _chroma.get_or_create_collection(
        name=col_name, embedding_function=_embed_fn,
        metadata={"file": filename, "file_hash": current_hash},
    )
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        col.add(
            documents=batch,
            ids=[f"{col_name}_{start + i}" for i in range(len(batch))],
            metadatas=[{"source": filename, "chunk_index": start + i} for i in range(len(batch))],
        )
    return {"status": "ingested", "collection": col_name, "chunks": len(chunks)}


def rag_ingest_all() -> List[dict]:
    results = []
    for doc in rag_list_docs():
        try:
            r = rag_ingest_doc(doc["filename"])
            results.append({**doc, **r})
        except Exception as e:
            log.error("Failed to ingest %s: %s", doc["filename"], e)
            results.append({**doc, "status": "error", "error": str(e), "chunks": 0})
    return results


def rag_retrieve(collection_id: str, query: str, n_results: int = None) -> List[str]:
    if n_results is None:
        n_results = cfg.rag_n_results

    ckey = _cache_key(collection_id, query)
    if ckey in _retrieval_cache:
        log.info("RAG cache hit for query hash %s", ckey[-8:])
        return _retrieval_cache[ckey]

    try:
        col = _chroma.get_collection(name=collection_id, embedding_function=_embed_fn)
    except Exception:
        return []
    count = col.count()
    if count == 0:
        return []
    results = col.query(query_texts=[query], n_results=min(n_results, count))
    chunks = results.get("documents", [[]])[0]

    if len(_retrieval_cache) >= _CACHE_MAX:
        oldest = next(iter(_retrieval_cache))
        del _retrieval_cache[oldest]
    _retrieval_cache[ckey] = chunks
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — LLM SERVICE  (streaming + non-streaming)
# ═══════════════════════════════════════════════════════════════════════════════

_openai = AsyncOpenAI(api_key=cfg.openai_api_key)

_SYSTEM_PROMPT = """You are Priya, a warm and knowledgeable property sales advisor for a trusted real estate platform.

YOUR PERSONALITY:
• Friendly and approachable — speak like a helpful friend, not a pushy salesperson
• Calm and reassuring — buying a home is a big decision; make the customer feel safe
• Honest and transparent — only share facts from the property documents; never exaggerate or invent details
• Enthusiastic but not over-the-top — show genuine excitement about great features

GREETING RULE:
• On the very FIRST message of a conversation (when chat_history is empty or absent), ALWAYS start with a warm greeting before answering. Example: "Hello! Welcome to [Property Name] — I'm so glad you reached out! 😊" or the equivalent in the user's language.
• On follow-up messages, skip the greeting and answer naturally.

ANSWERING RULES:
• Answer ONLY based on the PROPERTY CONTEXT provided below — do not invent prices, areas, amenities, or timelines
• If the answer is not in the context, say honestly: "That detail isn't in the brochure I have right now, but I'd love to find out for you!"
• Keep answers concise (3–5 sentences max) unless the customer asks for details
• End responses with a gentle, helpful follow-up question when appropriate (e.g., "Would you like to know about the payment plan?")
• Use light positive language: "Great question!", "Absolutely!", "You'll love this part…"

Always reply in the language specified in the language instruction below."""

_LANG_INSTRUCTION = {
    "english":  "Respond in English.",
    "hindi":    "Respond in Hindi (हिन्दी में जवाब दें).",
    "gujarati": "Respond in Gujarati (ગુજરાતીમાં જવાબ આપો).",
    "telugu":   "Respond in Telugu (తెలుగులో సమాధానం ఇవ్వండి).",
    "tamil":    "Respond in Tamil (தமிழில் பதில் அளிக்கவும்).",
}


def _build_messages(
    query: str,
    context_chunks: List[str],
    output_language: str,
    chat_history: Optional[List[dict]],
) -> list:
    context_str = (
        "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    )
    lang_inst = _LANG_INSTRUCTION.get(output_language.lower(), "Respond in English.")
    messages = [
        {
            "role": "system",
            "content": f"{_SYSTEM_PROMPT}\n\n{lang_inst}\n\nPROPERTY CONTEXT:\n{context_str}",
        }
    ]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": query})
    return messages


async def llm_answer(
    query: str,
    context_chunks: List[str],
    output_language: str = "english",
    chat_history: Optional[List[dict]] = None,
) -> str:
    messages = _build_messages(query, context_chunks, output_language, chat_history)
    resp = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


async def llm_stream(
    query: str,
    context_chunks: List[str],
    output_language: str = "english",
    chat_history: Optional[List[dict]] = None,
) -> AsyncIterator[str]:
    messages = _build_messages(query, context_chunks, output_language, chat_history)
    stream = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600,
        temperature=0.3,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SARVAM SERVICE  (parallel TTS + per-language female speakers)
# ═══════════════════════════════════════════════════════════════════════════════

_SARVAM_BASE = "https://api.sarvam.ai"

_LANG_CODE = {
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "gujarati": "gu-IN",
    "telugu":   "te-IN",
    "tamil":    "ta-IN",
}

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


async def sarvam_stt(audio_bytes: bytes, audio_format: str = "wav") -> str:
    headers = {"api-subscription-key": cfg.sarvam_api_key}
    files = {"file": (f"audio.{audio_format}", audio_bytes, f"audio/{audio_format}")}
    data  = {"model": "saaras:v3", "language_code": "unknown", "with_diarization": "false"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_SARVAM_BASE}/speech-to-text", headers=headers, files=files, data=data
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "")


def _split_tts(text: str, limit: int = 450) -> List[str]:
    sentences = re.split(r"(?<=[।.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= limit:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            while len(s) > limit:
                chunks.append(s[:limit])
                s = s[limit:]
            current = s
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


async def _tts_chunk(chunk_text: str, lang_code: str, speaker: str) -> bytes:
    headers = {
        "api-subscription-key": cfg.sarvam_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [chunk_text],
        "target_language_code": lang_code,
        "speaker": speaker,
        "model": cfg.sarvam_tts_model,
        "enable_preprocessing": True,
        "speech_sample_rate": 22050,
        "enc_format": "wav",
        "loudness_normalization": True,
    }
    client = get_http_client()
    resp = await client.post(f"{_SARVAM_BASE}/text-to-speech", headers=headers, json=payload)
    if resp.status_code != 200:
        log.error("Sarvam TTS error %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
    audios = resp.json().get("audios", [])
    if not audios:
        raise ValueError("No audio returned from Sarvam TTS")
    return base64.b64decode(audios[0])


async def sarvam_tts(text: str, language: str = "english") -> bytes:
    lang_code = _LANG_CODE.get(language.lower(), "en-IN")
    speaker   = _LANG_SPEAKER.get(language.lower(), "anushka")
    chunks    = _split_tts(text)
    log.info(
        "TTS: %d chunk(s) | language=%s | speaker=%s [parallel]",
        len(chunks), lang_code, speaker,
    )

    if len(chunks) == 1:
        return await _tts_chunk(chunks[0], lang_code, speaker)

    wav_parts: List[bytes] = await asyncio.gather(
        *[_tts_chunk(c, lang_code, speaker) for c in chunks]
    )

    first_wav       = wav_parts[0]
    sample_rate     = int.from_bytes(first_wav[24:28], "little")
    num_channels    = int.from_bytes(first_wav[22:24], "little")
    bits_per_sample = int.from_bytes(first_wav[34:36], "little")
    pcm_data        = b"".join(w[44:] for w in wav_parts)
    data_size       = len(pcm_data)
    byte_rate       = sample_rate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8

    header = (
        b"RIFF" + struct.pack("<I", 36 + data_size) +
        b"WAVE" +
        b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                              byte_rate, block_align, bits_per_sample) +
        b"data" + struct.pack("<I", data_size)
    )
    return header + pcm_data


async def sarvam_translate(text: str, target_language: str) -> str:
    if target_language.lower() == "english":
        return text
    lang_code = _LANG_CODE.get(target_language.lower(), "en-IN")
    if lang_code == "en-IN":
        return text
    headers = {
        "api-subscription-key": cfg.sarvam_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": lang_code,
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }
    try:
        client = get_http_client()
        resp = await client.post(f"{_SARVAM_BASE}/translate", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json().get("translated_text", text)
    except Exception as e:
        log.warning("Translation failed (%s): %s", target_language, e)
        return text


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Startup — ingesting documents from %s ...", cfg.rag_dir)
    Path(cfg.rag_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, rag_ingest_all)
    for r in results:
        log.info(
            "  %-45s → %-10s (%d chunks)",
            r.get("filename", "?"), r.get("status", "?"), r.get("chunks", 0),
        )
    yield
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    log.info("Shutdown complete.")


app = FastAPI(
    title="PropVoice — Real Estate RAG API",
    description="RAG chatbot with Sarvam STT/TTS for Indian real estate listings",
    version="2.4.0",
    lifespan=lifespan,
)

# ── CORS — fully open ─────────────────────────────────────────────────────────
# Any domain (including your frontend team's localhost, Vercel, Netlify, etc.)
# can call this API with no restrictions.
#
# NOTE: allow_credentials MUST stay False when allow_origins=["*"].
# If you ever need to send cookies or Authorization headers from a specific
# origin, replace ["*"] with ["https://yourfrontend.com"] and flip to True.
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
# Serves everything inside ./static/ at /static/*
# Also serves index.html at / (root) if it exists.
# The folder is created automatically so the app never crashes on startup
# even when the frontend hasn't been built yet.
# ─────────────────────────────────────────────────────────────────────────────
_static_dir = Path("./static")
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Pydantic schemas — API contract is UNCHANGED ─────────────────────────────

class ChatRequest(BaseModel):
    query: str                          = Field(...,        description="User question in any language")
    pdf_id: str                         = Field(...,        description="Collection ID from GET /api/pdfs")
    output_language: str                = Field("english",  description="english | hindi | gujarati | telugu | tamil")
    chat_history: Optional[List[dict]]  = Field(None,       description="Previous turns [{role,content}]")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the price of this flat?",
                "pdf_id": "saffron_heights",
                "output_language": "hindi",
                "chat_history": [],
            }
        }
    }


class ChatResponse(BaseModel):
    answer_text: str
    audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV audio of the answer")
    pdf_id: str
    output_language: str


class IngestResult(BaseModel):
    filename: str
    id: str
    status: str
    chunks: Optional[int] = None
    error: Optional[str]  = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve frontend index.html if present, otherwise return a JSON welcome."""
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {
        "message": "PropVoice API is running.",
        "docs": "/docs",
        "health": "/api/health",
        "properties": "/api/pdfs",
    }


@app.get("/api/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "2.4.0"}


@app.get("/api/pdfs", tags=["Properties"])
async def get_pdfs():
    docs = rag_list_docs()
    return {"pdfs": docs, "count": len(docs)}


@app.post("/api/ingest", tags=["Admin"])
async def ingest_all_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(rag_ingest_all)
    return {"message": "Ingestion started in background. Check server logs for progress."}


@app.post("/api/ingest/{pdf_id}", tags=["Admin"], response_model=IngestResult)
async def ingest_one_endpoint(pdf_id: str):
    docs = rag_list_docs()
    matched = next((d for d in docs if d["id"] == pdf_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"No file with id '{pdf_id}' found in RAG/ folder")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rag_ingest_doc, matched["filename"])
        return IngestResult(filename=matched["filename"], id=pdf_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(req: ChatRequest, stream: bool = False):
    """
    **Text → Text + Audio**

    Pass `?stream=true` for SSE streaming:
      - `event: token`  data: <word>
      - `event: audio`  data: <base64_wav>
      - `event: done`   data: [DONE]
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    chunks = await asyncio.get_event_loop().run_in_executor(
        None, rag_retrieve, req.pdf_id, req.query, cfg.rag_n_results
    )

    if stream:
        async def event_stream():
            full_answer_parts = []
            async for token in llm_stream(req.query, chunks, req.output_language, req.chat_history):
                full_answer_parts.append(token)
                yield f"event: token\ndata: {token}\n\n"

            full_answer = "".join(full_answer_parts)

            if req.output_language.lower() != "english":
                full_answer = await sarvam_translate(full_answer, req.output_language)

            try:
                wav_bytes = await sarvam_tts(full_answer, req.output_language)
                audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                yield f"event: audio\ndata: {audio_b64}\n\n"
            except Exception as e:
                log.warning("TTS failed in stream mode: %s", e)

            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    answer = await llm_answer(
        query=req.query,
        context_chunks=chunks,
        output_language=req.output_language,
        chat_history=req.chat_history,
    )

    if req.output_language.lower() != "english":
        answer = await sarvam_translate(answer, req.output_language)

    audio_b64: Optional[str] = None
    try:
        wav_bytes = await sarvam_tts(answer, req.output_language)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    except Exception as e:
        log.warning("TTS failed: %s", e)

    return ChatResponse(
        answer_text=answer,
        audio_base64=audio_b64,
        pdf_id=req.pdf_id,
        output_language=req.output_language,
    )


@app.post("/api/voice", response_model=ChatResponse, tags=["Chat"])
async def voice_endpoint(
    audio: UploadFile       = File(..., description="WAV/WebM audio file"),
    pdf_id: str             = Form(..., description="Property collection ID from GET /api/pdfs"),
    output_language: str    = Form("english", description="english | hindi | gujarati | telugu | tamil"),
):
    """**Voice → Text + Audio** — Sarvam Saaras v3 auto-detects Indian language."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        transcript = await sarvam_stt(audio_bytes, audio_format="wav")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sarvam STT error: {e}")

    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio — please speak clearly and try again",
        )

    log.info("STT transcript: %s", transcript[:120])

    chunks = await asyncio.get_event_loop().run_in_executor(
        None, rag_retrieve, pdf_id, transcript, cfg.rag_n_results
    )
    answer = await llm_answer(
        query=transcript,
        context_chunks=chunks,
        output_language=output_language,
    )

    if output_language.lower() != "english":
        answer = await sarvam_translate(answer, output_language)

    audio_b64: Optional[str] = None
    try:
        wav_bytes = await sarvam_tts(answer, output_language)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    except Exception as e:
        log.warning("TTS failed: %s", e)

    return ChatResponse(
        answer_text=answer,
        audio_base64=audio_b64,
        pdf_id=pdf_id,
        output_language=output_language,
    )


@app.delete("/api/cache", tags=["Admin"])
async def clear_cache():
    size = len(_retrieval_cache)
    _retrieval_cache.clear()
    return {"cleared_entries": size}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
        log_level="info",
    )