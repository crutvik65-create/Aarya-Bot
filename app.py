"""
PropVoice — Pacifica Companies RAG API  v6.0.0
Enterprise-grade: 5000 users/day, 5 languages, accurate TTS

FIXES vs v5:
  FIX-1  Intent classifier reordered: PRICE/LOCATION/CONTACT before PERSON.
         Added LOCATION intent. "price", "location", "Price sqft?" now route correctly.
  FIX-2  "price range" across all projects now hits COMPARE path (added
         "price range" / "range" keywords). Multi-collection PRICE queries
         fall back to COMPARE so the LLM sees all pricing chunks.
  FIX-3  Token optimisation: small-doc threshold raised to 15 (was 10).
         Meta-filter results capped per collection. Embedding already uses
         OpenAI text-embedding-3-small via _get_query_embedding() with LRU
         cache — no change needed there.
  FIX-4  Pincode / area / sqft TTS: added PINCODE rule in all 5 language
         TTS blocks. 6-digit address numbers → digit-by-digit. 3-4 digit
         area/sqft numbers stay as cardinal words (not treated as currency).
         Prices stay as lakh/crore words.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

os.environ["ANONYMIZED_TELEMETRY"]        = "false"
os.environ["CHROMA_TELEMETRY"]            = "false"
os.environ["CHROMA_CLIENT_AUTH_PROVIDER"] = ""

import chromadb
try:
    import chromadb.telemetry.product.posthog as _ph
    _ph.Posthog.capture = lambda self, *a, **kw: None
except Exception:
    pass

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("propvoice")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    openai_api_key:     str = Field(...,            env="OPENAI_API_KEY")
    sarvam_api_key:     str = Field(...,            env="SARVAM_API_KEY")
    chroma_persist_dir: str = Field("./chroma_db",  env="CHROMA_PERSIST_DIR")
    rag_dir:            str = Field("./RAG",         env="RAG_DIR")
    host:               str = Field("0.0.0.0",       env="HOST")
    port:               int = Field(8000,             env="PORT")
    workers:            int = Field(1,                env="WORKERS")
    sarvam_tts_model:   str = Field("bulbul:v3",     env="SARVAM_TTS_MODEL")
    rag_n_results:      int = Field(6,                env="RAG_N_RESULTS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

cfg = Settings()

_LANG_SPEAKER = {
    "english":  os.getenv("SARVAM_SPEAKER_ENGLISH",  "ritu"),
    "hindi":    os.getenv("SARVAM_SPEAKER_HINDI",     "ritu"),
    "gujarati": os.getenv("SARVAM_SPEAKER_GUJARATI",  "ritu"),
    "telugu":   os.getenv("SARVAM_SPEAKER_TELUGU",    "ritu"),
    "tamil":    os.getenv("SARVAM_SPEAKER_TAMIL",     "ritu"),
}
_LANG_CODE = {
    "english":  "en-IN",
    "hindi":    "hi-IN",
    "gujarati": "gu-IN",
    "telugu":   "te-IN",
    "tamil":    "ta-IN",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — SQLITE CONVERSATION STORE
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "./conversations.db"

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _db_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        session_id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
        phone TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'english',
        properties TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL)""")
    conn.commit(); conn.close()

def db_upsert_user(session_id: str, name: str = "", phone: str = ""):
    conn = _db_conn()
    conn.execute("""INSERT INTO users (session_id, name, phone, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            name  = CASE WHEN excluded.name  != '' THEN excluded.name  ELSE name  END,
            phone = CASE WHEN excluded.phone != '' THEN excluded.phone ELSE phone END
    """, (session_id, name, phone, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_get_user(session_id: str) -> dict:
    conn = _db_conn()
    row = conn.execute("SELECT * FROM users WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}

def db_save_turn(session_id: str, role: str, content: str,
                 language: str = "english", properties: str = ""):
    conn = _db_conn()
    conn.execute("""INSERT INTO conversations (session_id,role,content,language,properties,ts)
        VALUES (?,?,?,?,?,?)""",
        (session_id, role, content, language, properties, datetime.now().isoformat()))
    conn.commit(); conn.close()

def db_get_history(session_id: str, limit: int = 10) -> List[dict]:
    conn = _db_conn()
    rows = conn.execute("""SELECT role, content FROM conversations
        WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
        (session_id, limit)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — DOCX EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

WNS_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", WNS_URI)

def extract_docx_text(file_path: Path) -> str:
    with zipfile.ZipFile(str(file_path), "r") as zf:
        with zf.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
    body = root.find(f".//{{{WNS_URI}}}body")
    if body is None:
        return ""
    sections = []
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "tbl":
            rows = []
            for row in child.iter(f"{{{WNS_URI}}}tr"):
                cells = []
                for cell in row.iter(f"{{{WNS_URI}}}tc"):
                    text = " ".join(
                        (n.text or "").strip() for n in cell.iter(f"{{{WNS_URI}}}t")
                        if (n.text or "").strip())
                    if text:
                        cells.append(text)
                if len(cells) == 2:
                    rows.append(f"{cells[0]}: {cells[1]}")
                elif len(cells) == 1:
                    rows.append(f"\n## {cells[0]}")
                elif len(cells) > 2:
                    rows.append(" | ".join(cells))
            if rows:
                sections.append("\n".join(rows))
        elif tag == "p":
            text = " ".join(
                (n.text or "").strip() for n in child.iter(f"{{{WNS_URI}}}t")
                if (n.text or "").strip())
            if text.strip():
                sections.append(text.strip())
    return "\n\n".join(s for s in sections if s.strip())

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — CHROMADB + RAG
# ─────────────────────────────────────────────────────────────────────────────

_chroma = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
_embed  = OpenAIEmbeddingFunction(api_key=cfg.openai_api_key,
                                   model_name="text-embedding-3-small")
_embed_cache: dict = {}
_EMBED_CACHE_MAX = 512          # bumped for 5k users/day

import openai as _openai_sync
_openai_sync_client = _openai_sync.OpenAI(api_key=cfg.openai_api_key)

def _col_name(filename: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name + "_col")[:63]

# ── Sentence-aware chunker ────────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'(?<=[.!?।])\s+(?=[A-Z"\u0A00-\u0AFF\u0900-\u097F])')

def _chunk_text(text: str, chunk_size: int = 150, overlap_sents: int = 2) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        sents = _SENT_SPLIT.split(para)
        sentences.extend(s.strip() for s in sents if s.strip())

    chunks: List[str] = []
    current_words: List[str] = []
    current_sents: List[str] = []

    def _flush():
        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)

    for sent in sentences:
        words = sent.split()
        if len(current_words) + len(words) > chunk_size and current_words:
            _flush()
            overlap = current_sents[-overlap_sents:] if overlap_sents else []
            current_words = [w for s in overlap for w in s.split()]
            current_sents = list(overlap)
        current_words.extend(words)
        current_sents.append(sent)

    _flush()
    return chunks

# ── Metadata tagger ───────────────────────────────────────────────────────────
_PERSON_PAT  = re.compile(r'\b(Mr\.|Ms\.|Mrs\.|Dr\.|Director|Manager|CEO|MD|President|Head|Officer|VP|Founder|Partner|[A-Z][a-z]+ [A-Z][a-z]+)\b')
_PRICE_PAT   = re.compile(r'(price|cost|rate|lakh|crore|Rs\.|₹|starting|sqft|sq\.?\s*ft|bhk|flat|unit|carpet|area|per\s+sq|onwards|pricing)', re.IGNORECASE)
_RERA_PAT    = re.compile(r'\bPR/[A-Z]{2}/', re.IGNORECASE)
_CONTACT_PAT = re.compile(r'(\b\d{10}\b|@|phone|email|contact|call|reach|whatsapp)', re.IGNORECASE)
_AMENITY_PAT = re.compile(r'(amenity|amenities|club|pool|gym|garden|parking|security|lift|elevator|play|lounge|spa|terrace)', re.IGNORECASE)
# FIX-1: add location tagger for chunk metadata
_LOCATION_PAT = re.compile(r'(location|address|situated|located|road|junction|sector|nagar|township|pincode|pin\s*code|\b\d{6}\b)', re.IGNORECASE)

def _tag_chunk(chunk: str) -> dict:
    return {
        "has_person":   int(bool(_PERSON_PAT.search(chunk))),
        "has_price":    int(bool(_PRICE_PAT.search(chunk))),
        "has_rera":     int(bool(_RERA_PAT.search(chunk))),
        "has_contact":  int(bool(_CONTACT_PAT.search(chunk))),
        "has_amenity":  int(bool(_AMENITY_PAT.search(chunk))),
        "has_location": int(bool(_LOCATION_PAT.search(chunk))),  # FIX-1
    }

def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()

def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".docx", ".doc"):
        try:
            return extract_docx_text(file_path)
        except zipfile.BadZipFile:
            raw = file_path.read_bytes()
            chunks = re.findall(rb"[\x20-\x7E]{4,}", raw)
            return " ".join(c.decode("ascii", errors="ignore") for c in chunks)
    raise ValueError(f"Unsupported: {suffix}")

_SUPPORTED = {".pdf", ".docx", ".doc"}

def rag_list_docs() -> List[dict]:
    rag_dir = Path(cfg.rag_dir)
    if not rag_dir.exists():
        return []
    files = sorted([p for p in rag_dir.glob("*")
                    if p.suffix.lower() in _SUPPORTED
                    and not p.name.startswith(("~$", "~"))])
    return [{"id": _col_name(p.name), "filename": p.name,
             "display_name": p.stem.replace("_"," ").replace("-"," ").title()}
            for p in files]

def rag_ingest_doc(filename: str) -> dict:
    doc_path     = Path(cfg.rag_dir) / filename
    col_name     = _col_name(filename)
    current_hash = _file_md5(str(doc_path))
    try:
        col = _chroma.get_collection(name=col_name, embedding_function=_embed)
        # FIX-1: force re-ingest if schema version changed (added has_location)
        schema_ver = col.metadata.get("schema_version", "0")
        if (col.metadata.get("file_hash") == current_hash
                and col.count() > 0
                and schema_ver == "2"):
            return {"status": "skipped", "collection": col_name, "chunks": col.count()}
        _chroma.delete_collection(col_name)
    except Exception:
        pass
    full_text = _extract_text(doc_path)
    if not full_text.strip():
        log.warning("No text extracted from %s", filename)
        col = _chroma.get_or_create_collection(
            name=col_name, embedding_function=_embed,
            metadata={"file": filename, "file_hash": current_hash, "schema_version": "2"})
        return {"status": "empty", "collection": col_name, "chunks": 0}
    chunks = _chunk_text(full_text)
    col = _chroma.get_or_create_collection(
        name=col_name, embedding_function=_embed,
        metadata={"file": filename, "file_hash": current_hash, "schema_version": "2"})
    for start in range(0, len(chunks), 100):
        batch = chunks[start: start + 100]
        metadatas = []
        for i, c in enumerate(batch):
            m = _tag_chunk(c)
            m["source"] = filename
            m["chunk_index"] = start + i
            metadatas.append(m)
        col.add(
            documents=batch,
            ids=[f"{col_name}_{start+i}" for i in range(len(batch))],
            metadatas=metadatas)
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

def _get_query_embedding(query: str) -> list:
    """LRU-cached embedding — single OpenAI call per unique query."""
    key = hashlib.sha256(query.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    resp   = _openai_sync_client.embeddings.create(input=[query], model="text-embedding-3-small")
    vector = resp.data[0].embedding
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        # evict oldest
        del _embed_cache[next(iter(_embed_cache))]
    _embed_cache[key] = vector
    return vector

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4B — QUERY INTENT CLASSIFIER  (FIX-1, FIX-2)
#
#  BUG ROOT CAUSE: In v5, PERSON check ran BEFORE PRICE/LOCATION checks.
#  "price and carpet area?" matched [A-Z][a-z]+ [A-Z][a-z]+ on ... actually
#  it was matching because "price" contains no capitals so that's not it.
#  The real cause: _PRICE_PAT_Q was checked AFTER _PERSON_PAT_Q.
#  "location details" → PERSON because "details" has no cap letter after space,
#  but the regex is case-insensitive so [A-Z][a-z]+ matches "location" followed
#  by a space... Actually on review: Python re.IGNORECASE on [A-Z][a-z]+
#  still only matches uppercase-first because \b[A-Z] is case-sensitive even
#  with IGNORECASE for character classes. The real issue for "location" → GENERAL
#  was that Aavaas_Hyderabad_col was selected and its 3 chunks didn't have location
#  data tagged (schema_version "1" had no has_location). For "price range" → the
#  query matches _PERSON_PAT_Q because "price" doesn't, "range" doesn't... wait.
#  Actually "What is the price range?" → log shows INTENT: PERSON. Checking:
#  _PERSON_PAT_Q pattern includes [A-Z][a-z]+\s+[A-Z][a-z]+ but "price range"
#  is all lowercase. So it must be matching something else. "What" + " is" = no.
#  Looking at the pattern again: it's `re.IGNORECASE` so [A-Z][a-z]+ with
#  IGNORECASE matches ANY letter followed by lowercase — so "pr ice" → no,
#  but "price" = p-r-i-c-e, that's 5 chars all matching [a-z] with ignorecase,
#  plus the \b word boundary... \b(who\s+is | ... | [A-Z][a-z]+\s+[A-Z][a-z]+)
#  WITH re.IGNORECASE means [A-Z] matches any letter. So "price range" matches
#  [A-Z][a-z]+\s+[A-Z][a-z]+ !! That's the exact bug. The PERSON pattern's
#  last clause is a catch-all for "two capitalized words" but with IGNORECASE
#  it matches ANY two words. FIX: remove IGNORECASE from _PERSON_PAT_Q OR
#  make the two-word pattern require actual uppercase first letter.
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_PAT = re.compile(
    r'^\s*(hi|hello|hey|thanks|thank you|okay|ok|sure|bye|goodbye|good\s*(morning|evening|afternoon|night)'
    r'|namaste|kem\s*cho|kem\s*chho|kaise\s*ho|हाय|हेलो|नमस्ते|ઠીક|ઓ\s*કે|ઓ\s*કેy|'
    r'__GREETING__|__ASK_PHONE__|__PHONE_SKIP__|__PHONE_DONE__)\s*[!?.]*\s*$',
    re.IGNORECASE)

# FIX-1: PERSON pattern — removed re.IGNORECASE so [A-Z][a-z]+ requires actual caps
# This prevents "price range", "carpet area", "location details" from matching
_PERSON_PAT_Q = re.compile(
    r'\b(who\s+is|who\'?s|contact\s+of|meet|director|manager|ceo|founder|head\s+of'
    r'|Mr\.|Ms\.|Dr\.|[A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    # NOTE: NO re.IGNORECASE here — intentional, two-word pattern needs real Title Case
)

_CONTACT_PAT_Q = re.compile(
    r'\b(phone|number|contact|call|email|reach|whatsapp|helpline|mobile)\b', re.IGNORECASE)

_RERA_PAT_Q = re.compile(
    r'\b(rera|registration|registered|pr/)\b', re.IGNORECASE)

_AMENITY_PAT_Q = re.compile(
    r'\b(amenity|amenities|facilities|club|pool|gym|garden|playground|security|lift|parking|spa)\b',
    re.IGNORECASE)

# FIX-1: new LOCATION intent
_LOCATION_PAT_Q = re.compile(
    r'\b(location|address|where|situated|located|road|junction|sector|nagar|landmark|'
    r'near|distance|km\b|miles?|pincode|pin\s*code|directions?|map|area|locality|'
    r'city|town|village|hyderabad|ahmedabad|bangalore|mumbai|pune|delhi)\b',
    re.IGNORECASE)

# FIX-2: added "price range", "range", "all price", "price list" to COMPARE
_COMPARE_PAT = re.compile(
    r'\b(compar|vs\.?|versus|difference|better|best|worst|cheapest|most\s+expensive|'
    r'lowest|highest|minimum|maximum|least|rank|which one|all sites?|all projects?|'
    r'all propert|across all|among all|between|price list|price range|full list|'
    r'starting price|all prices?|list\s+all|show\s+all|overview)\b'
    r'|(સૌથી|બધી|सबसे|सभी|तुलना|সব|সবচেয়ে)',
    re.IGNORECASE)

_PRICE_PAT_Q = re.compile(
    r'\b(price|cost|rate|budget|afford|lakh|crore|cheap|expensive|how much|what.*cost|'
    r'bhk.*price|price.*bhk|starting|onwards|sqft|sq\s*ft|carpet|per\s+sq|per\s+foot|'
    r'psf|pricing|range)\b', re.IGNORECASE)

QueryIntent = str

def classify_query(query: str) -> QueryIntent:
    """
    Lightweight rule-based classifier — zero LLM cost.

    PRIORITY ORDER (most specific → most general):
      SKIP → COMPARE → PRICE → LOCATION → CONTACT → RERA → AMENITY → PERSON → GENERAL

    PERSON is intentionally checked LAST because its two-word Title Case pattern
    can false-positive on property names embedded in other query types.
    """
    q = query.strip()
    if _SKIP_PAT.match(q):
        return "SKIP"
    if _COMPARE_PAT.search(q):
        return "COMPARE"
    if _PRICE_PAT_Q.search(q):          # FIX-1: PRICE before PERSON
        return "PRICE"
    if _LOCATION_PAT_Q.search(q):       # FIX-1: new LOCATION intent
        return "LOCATION"
    if _CONTACT_PAT_Q.search(q):
        return "CONTACT"
    if _RERA_PAT_Q.search(q):
        return "RERA"
    if _AMENITY_PAT_Q.search(q):
        return "AMENITY"
    if _PERSON_PAT_Q.search(q):         # FIX-1: PERSON last
        return "PERSON"
    return "GENERAL"

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4C — SMART RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

# FIX-3: raised small-doc threshold from 10 → 15
_SMALL_DOC_THRESHOLD = 15

def _meta_filter_fetch(col_name: str, flag: str, semantic_fallback_query: str = "",
                       n: int = None, cap: int = 8) -> List[str]:
    """
    Fetch chunks by metadata flag. Hard-cap results to avoid token explosion.
    Falls back to semantic search if no metadata hits.
    cap: maximum chunks returned (FIX-3: enterprise token control)
    """
    if n is None:
        n = cfg.rag_n_results
    try:
        col   = _chroma.get_collection(name=col_name)
        count = col.count()
        if count == 0:
            return []
        res  = col.get(where={flag: {"$eq": 1}}, include=["documents"])
        docs = res.get("documents", [])
        if docs:
            log.info("META-FILTER %s [%s]: found %d chunks (cap=%d)", col_name, flag, len(docs), cap)
            return docs[:cap]
        if semantic_fallback_query:
            vector = _get_query_embedding(semantic_fallback_query)
            n_use  = min(n, count)
            res2   = col.query(query_embeddings=[vector], n_results=n_use)
            fallback = res2.get("documents", [[]])[0]
            log.info("META-FILTER fallback semantic %s: %d chunks", col_name, len(fallback))
            return fallback
    except Exception as e:
        log.warning("Meta-filter failed %s [%s]: %s", col_name, flag, e)
    return []

def rag_retrieve(collection_id: str, query: str, intent: QueryIntent = "GENERAL") -> List[str]:
    """
    Single-collection retrieval. Intent-aware.
    FIX-3: small-doc threshold raised to 15.
    FIX-1: LOCATION intent maps to has_location flag.
    FIX-2: PRICE on single collection also uses has_price filter with semantic fallback.
    """
    try:
        col   = _chroma.get_collection(name=collection_id)
        count = col.count()
    except Exception:
        return []
    if count == 0:
        return []

    # Small doc: return all (cheap, avoids any retrieval miss)
    if count <= _SMALL_DOC_THRESHOLD:
        res    = col.get(include=["documents"])
        chunks = res.get("documents", [])
        log.info("RAG full-fetch %s (%d chunks, small doc)", collection_id, len(chunks))
        return chunks

    intent_to_flag = {
        "PERSON":   "has_person",
        "CONTACT":  "has_contact",
        "RERA":     "has_rera",
        "AMENITY":  "has_amenity",
        "PRICE":    "has_price",
        "LOCATION": "has_location",   # FIX-1
    }
    if intent in intent_to_flag:
        return _meta_filter_fetch(
            collection_id, intent_to_flag[intent],
            semantic_fallback_query=query,
            n=cfg.rag_n_results,
            cap=10)

    # GENERAL / COMPARE: semantic search
    vector = _get_query_embedding(query)
    n_use  = min(cfg.rag_n_results, count)
    res    = col.query(query_embeddings=[vector], n_results=n_use)
    chunks = res.get("documents", [[]])[0]
    log.info("RAG semantic %s: %d chunks (intent=%s)", collection_id, len(chunks), intent)
    return chunks

def _extract_pricing_chunks(col_name: str, cap: int = 4) -> List[str]:
    """For COMPARE queries: fetch price-tagged chunks, capped tightly."""
    try:
        col   = _chroma.get_collection(name=col_name)
        count = col.count()
        if count == 0:
            return []
        res   = col.get(where={"has_price": {"$eq": 1}}, include=["documents"])
        docs  = res.get("documents", [])
        if docs:
            return docs[:cap]
        res2  = col.get(include=["documents"])
        return res2.get("documents", [])[:3]
    except Exception as e:
        log.warning("Price chunk extract failed for %s: %s", col_name, e)
        return []

def rag_retrieve_multi(collection_ids: List[str], query: str,
                       intent: QueryIntent = "GENERAL") -> dict:
    """
    Multi-collection retrieval. Intent-aware, token-controlled.

    FIX-2: When intent=PRICE and >1 collection, treat as COMPARE to ensure
    all properties return price chunks. "price range for all" needs this.
    """
    # FIX-2: PRICE on multi-collection → treat like COMPARE
    effective_intent = intent
    if intent == "PRICE" and len(collection_ids) > 1:
        effective_intent = "COMPARE"
        log.info("RAG multi: PRICE on %d collections → promoting to COMPARE", len(collection_ids))

    is_comparison = (effective_intent == "COMPARE")

    intent_to_flag = {
        "PERSON":   "has_person",
        "CONTACT":  "has_contact",
        "RERA":     "has_rera",
        "AMENITY":  "has_amenity",
        "LOCATION": "has_location",   # FIX-1
    }

    vector = None
    if not is_comparison and effective_intent not in intent_to_flag:
        vector = _get_query_embedding(query)

    def _query_one(cid: str) -> tuple:
        if is_comparison:
            return cid, _extract_pricing_chunks(cid, cap=4)
        if effective_intent in intent_to_flag:
            # FIX-3: cap at 5 per collection for multi-collection metadata queries
            chunks = _meta_filter_fetch(
                cid, intent_to_flag[effective_intent],
                semantic_fallback_query=query, n=4, cap=5)
            return cid, chunks
        # GENERAL: semantic, capped
        try:
            col   = _chroma.get_collection(name=cid)
            count = col.count()
            if count == 0:
                return cid, []
            # Small doc: return all
            if count <= _SMALL_DOC_THRESHOLD:
                res    = col.get(include=["documents"])
                return cid, res.get("documents", [])
            n_use = min(cfg.rag_n_results, count)
            if vector is None:
                v = _get_query_embedding(query)
            else:
                v = vector
            res   = col.query(query_embeddings=[v], n_results=n_use)
            chunks = res.get("documents", [[]])[0]
            return cid, chunks
        except Exception as e:
            log.warning("RAG multi-retrieve failed for %s: %s", cid, e)
            return cid, []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(collection_ids), 10)) as executor:
        futures = {executor.submit(_query_one, cid): cid for cid in collection_ids}
        for future in as_completed(futures):
            cid, chunks = future.result()
            if chunks:
                results[cid] = chunks
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — RERA NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

_BROKEN_RERA_PAT = re.compile(
    r'(?<![A-Za-z0-9])([A-Z]\s+[A-Z])\s*,\s*([A-Z]\s+[A-Z])\s*,\s*'
    r'([A-Za-z0-9][A-Za-z0-9 ]*?(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9 ]*?){2,8})'
    r'(?=[।\.\!\?\n"\u0A00-\u0AFF\u0900-\u097F]|$)',
    re.IGNORECASE)

def _fix_broken_rera(m: re.Match) -> str:
    parts  = re.split(r'\s*,\s*', m.group(0))
    tokens = [re.sub(r'\s+', '', p.strip()).upper() for p in parts if p.strip()]
    return '/'.join(t for t in tokens if t)

def normalise_rera(text: str) -> str:
    return _BROKEN_RERA_PAT.sub(_fix_broken_rera, text)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — LLM
# ─────────────────────────────────────────────────────────────────────────────

_openai = AsyncOpenAI(api_key=cfg.openai_api_key)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6A — TTS NUMBER RULES  (FIX-4)
#
#  KEY ADDITIONS vs v5:
#  1. PINCODE rule: 6-digit standalone number in an address context → digit-by-digit
#  2. AREA/SQFT rule: 3-4 digit number followed by sq ft / sqft / ચો.ફૂ. → cardinal words
#     (e.g. 802 sq ft → eight hundred two square feet — NOT "eight zero two")
#     This was already correct in v5 but now EXPLICITLY stated to avoid confusion
#  3. MOBILE rule: 10-digit number → digit-by-digit (was already there, now explicit)
#  4. PRICE rule: number followed by lakh/crore → spoken as cardinal + unit word
#  5. NEVER treat 6-digit pincode as a monetary amount (the 500049 → "five lakh" bug)
#
#  DECISION TREE for any number in TTS field:
#    Is it after "Rs." / "₹" / followed by "lakh"/"crore"?  → currency words
#    Is it 10 digits?                                         → digit-by-digit (mobile)
#    Is it 6 digits standalone in an address?                 → digit-by-digit (pincode)
#    Is it a RERA segment?                                    → per RERA rules
#    Is it 3-4 digits followed by sq ft / sqft?              → cardinal words (area)
#    Otherwise                                                → cardinal words
# ─────────────────────────────────────────────────────────────────────────────

_TTS_NUMBER_RULES = {

"english": """\
TTS FIELD RULES — English
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN RULE: "tts" is the spoken version of "ui". Every number must match exactly.

NUMBER TYPE DECISION TREE (apply top-to-bottom):
  1. PRICE  — number before/after "lakh" or "crore" or after "Rs."/"₹"
              → speak as currency words
              44 lakh → forty four lakh | 64.99 lakh → sixty four point nine nine lakh
  2. MOBILE — exactly 10 consecutive digits
              → digit by digit
              9876543210 → nine eight seven six five four three two one zero
  3. PINCODE — exactly 6 digits appearing in an address (after city name, or hyphenated like "Hyderabad-500049")
              → ALWAYS digit by digit — NEVER as a monetary amount
              500049 → five zero zero zero four nine
              400001 → four zero zero zero zero one
              ⚠ COMMON MISTAKE: 500049 is NOT "five lakh forty nine" — it is a PINCODE
  4. RERA segment — part of a RERA code (split by "/"):
              Rule A: segment has ONLY letters AND length > 5 → speak as word
              Rule B: ≤5 chars OR has any digit → spell every character
              PR/GJ/GANDHINAGAR/AUDA/RAA04324/A1R/211021 →
                "P R, G J, Gandhinagar, A U D A, R A A zero four three two four, A one R, two one one zero two one"
  5. AREA   — 3-4 digit number followed by "sq ft", "sqft", "sq.ft", "square feet"
              → cardinal words + "square feet"
              802 sq ft → eight hundred two square feet
              575 sq ft → five hundred seventy five square feet
  6. DEFAULT — all other numbers → cardinal words
              44 → forty four | 85 → eighty five | 64 → sixty four

PRICES (full examples):
  Rs. 44 lakhs     → rupees forty four lakhs
  Rs. 64.99 lakhs  → rupees sixty four point nine nine lakhs
  Rs. 1.20 Crores  → rupees one point two zero crores
  ⚠ NEVER say "rupees five lakh forty nine" for pincode 500049 — that is WRONG

BHK:  2 BHK → two BHK  |  3 BHK → three BHK  |  4 BHK → four BHK
""",

"hindi": """\
TTS FIELD RULES — हिंदी
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
सुनहरा नियम: "tts" field "ui" का बोला जाने वाला रूप है।

संख्या प्रकार निर्णय वृक्ष (ऊपर से नीचे लागू करें):
  1. मूल्य — "लाख"/"करोड़"/"रुपये" के साथ संख्या → हिंदी में मूल्य शब्द
              44 लाख → चौवालीस लाख | 64.99 लाख → चौंसठ दशमलव नौ नौ लाख
  2. मोबाइल — ठीक 10 अंक → एक-एक अंक बोलें
              9876543210 → नौ आठ सात छह पाँच चार तीन दो एक शून्य
  3. पिनकोड — 6 अंक जो पते में हों (शहर के नाम के बाद या हाइफन के बाद)
              → हमेशा एक-एक अंक — कभी मौद्रिक राशि के रूप में नहीं
              500049 → पाँच शून्य शून्य शून्य चार नौ
              400001 → चार शून्य शून्य शून्य शून्य एक
              ⚠ गलती: 500049 को "पाँच लाख उनचास" मत कहें — यह पिनकोड है
  4. RERA — "/" से विभाजित segment:
              Rule A: केवल letters और length > 5 → शब्द के रूप में
              Rule B: ≤5 chars या कोई digit → एक-एक character (English में)
  5. क्षेत्रफल — 3-4 अंक + "वर्ग फुट" → हिंदी में गणना शब्द
              802 वर्ग फुट → आठ सौ दो वर्ग फुट
  6. अन्य — cardinal शब्द
              44 → चौवालीस | 85 → पचासी | 50 → पचास

मूल्य (पूर्ण उदाहरण):
  Rs. 44 लाख → रुपये चौवालीस लाख
  Rs. 85 लाख → रुपये पचासी लाख ← "पचास लाख" नहीं
  Rs. 64.99 लाख → रुपये चौंसठ दशमलव नौ नौ लाख

BHK:  2 BHK → दो BHK  |  3 BHK → तीन BHK  |  4 BHK → चार BHK
""",

"gujarati": """\
TTS FIELD RULES — ગુજરાતી
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
સોનેરી નિયમ: "tts" field એ "ui" નું બોલાયેલ રૂપ છે.

સંખ્યા પ્રકાર નિર્ણય વૃક્ષ (ઉપરથી નીચે અનુસરો):
  1. ભાવ — "લાખ"/"કરોડ"/"રૂપિયા" સાથેની સંખ્યા → ગુજરાતી ભાવ શબ્દો
              44 → ચુમ્માળીસ | 50 → પચાસ | 85 → પંચ્યાસી | 64.99 → ચોસઠ દશાંશ નવ નવ
  2. મોબાઈલ — બરાબર 10 અંક → એક-એક અંક
              9876543210 → નવ આઠ સાત છ પાંચ ચાર ત્રણ બે એક શૂન્ય
  3. પિનકોડ — 6 અંક જે સરનામામાં હોય (શહેર પછી અથવા hyphen પછી)
              → હંમેશા એક-એક અંક — ક્યારેય નાણાકીય રાશિ તરીકે નહીં
              500049 → પાંચ શૂન્ય શૂન્ય શૂન્ય ચાર નવ
              400001 → ચાર શૂન્ય શૂન્ય શૂન્ય શૂન્ય એક
              380006 → ત્રણ આઠ શૂન્ય શૂન્ય છ
              ⚠ ભૂલ: 500049 ને "પાંચ લાખ ઓગણ પચાસ" ન કહો — આ પિનકોડ છે
  4. RERA — "/" segment:
              Rule A: ફક્ત letters અને length > 5 → શબ્દ
              Rule B: ≤5 chars અથવા કોઈ digit → character-by-character (English)
  5. ક્ષેત્ર — 3-4 અંક + "ચો.ફૂ."/"ચોરસ ફૂટ"/"sq ft" → ગુજરાતી ગાણ શબ્દ
              802 ચો.ફૂ. → આઠ સો બે ચોરસ ફૂટ
              575 ચો.ફૂ. → પાંચ સો પંચોતેર ચોરસ ફૂટ
              588 ચો.ફૂ. → પાંચ સો અઠ્ઠ્યાસી ચોરસ ફૂટ
              774 ચો.ફૂ. → સાત સો ચુમ્યોતેર ચોરસ ફૂટ
              808 ચો.ફૂ. → આઠ સો આઠ ચોરસ ફૂટ
  6. અન્ય → ગુજરાતી cardinal શબ્દ

ભાવ (સંપૂર્ણ ઉદાહરણ):
  Rs. 44 લાખ → રૂપિયા ચુમ્માળીસ લાખ
  Rs. 50 લાખ → રૂપિયા પચાસ લાખ
  Rs. 85 લાખ → રૂપિયા પંચ્યાસી લાખ ← "પચાસ" નહીં — "પંચ્યાસી"
  Rs. 64.99 લાખ → રૂપિયા ચોસઠ દશાંશ નવ નવ લાખ

BHK:  2 BHK → બે BHK  |  3 BHK → ત્રણ BHK  |  4 BHK → ચાર BHK
""",

"telugu": """\
TTS FIELD RULES — తెలుగు
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
గోల్డెన్ రూల్: "tts" field అనేది "ui" యొక్క మాట్లాడే రూపం.

సంఖ్య రకాల నిర్ణయ వృక్షం (పైనుండి కిందికి అనుసరించండి):
  1. ధర — "లక్షలు"/"కోట్లు"/"రూపాయలు" తో సంఖ్య → తెలుగు ధర పదాలు
              44 → నలభై నాలుగు | 50 → యాభై | 85 → ఎనభై అయిదు | 64.99 → అరవై నాలుగు దశాంశం తొమ్మిది తొమ్మిది
  2. మొబైల్ — సరిగ్గా 10 అంకెలు → ఒక్కో అంకె
              9876543210 → తొమ్మిది ఎనిమిది ఏడు ఆరు అయిదు నాలుగు మూడు రెండు ఒకటి సున్న
  3. పిన్‌కోడ్ — చిరునామాలో 6 అంకెల సంఖ్య (నగరం తర్వాత లేదా హైఫన్ తర్వాత)
              → ఎల్లప్పుడూ ఒక్కో అంకె — ఎప్పుడూ ద్రవ్య మొత్తంగా చెప్పవద్దు
              500049 → అయిదు సున్న సున్న సున్న నాలుగు తొమ్మిది
              ⚠ తప్పు: 500049 ను "అయిదు లక్షల నలభై తొమ్మిది" అని చెప్పవద్దు
  4. RERA — "/" segment:
              Rule A: అక్షరాలు మాత్రమే మరియు length > 5 → పదంగా
              Rule B: ≤5 chars లేదా digit ఉంటే → character-by-character (English)
  5. విస్తీర్ణం — 3-4 అంకెలు + "చ.అ."/"చదరపు అడుగులు"/"sq ft" → తెలుగు cardinal పదాలు
              802 చ.అ. → ఎనిమిది వందల రెండు చదరపు అడుగులు
  6. ఇతరాలు → తెలుగు cardinal పదాలు

ధర (పూర్తి ఉదాహరణలు):
  Rs. 44 లక్షలు → రూపాయలు నలభై నాలుగు లక్షలు
  Rs. 85 లక్షలు → రూపాయలు ఎనభై అయిదు లక్షలు ← "యాభై" కాదు
  Rs. 64.99 లక్షలు → రూపాయలు అరవై నాలుగు దశాంశం తొమ్మిది తొమ్మిది లక్షలు

BHK:  2 BHK → రెండు BHK  |  3 BHK → మూడు BHK  |  4 BHK → నాలుగు BHK
""",

"tamil": """\
TTS FIELD RULES — தமிழ்
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
தங்கவிதி: "tts" field என்பது "ui" இன் பேசப்படும் வடிவம்.

எண் வகை முடிவு மரம் (மேலிருந்து கீழாக பின்பற்றுங்கள்):
  1. விலை — "லட்சம்"/"கோடி"/"ரூபாய்" உடன் எண் → தமிழ் விலை வார்த்தைகள்
              44 → நாற்பத்து நான்கு | 50 → ஐம்பது | 85 → எண்பத்தைந்து
  2. மொபைல் — சரியாக 10 இலக்கங்கள் → ஒவ்வொரு இலக்கமாக
              9876543210 → ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று பூஜ்யம்
  3. பின்கோடு — முகவரியில் 6 இலக்க எண் (நகரத்திற்கு பிறகு அல்லது ஹைஃபன் பிறகு)
              → எப்போதும் ஒவ்வொரு இலக்கமாக — நிதி தொகையாக சொல்லவே வேண்டாம்
              500049 → ஐந்து பூஜ்யம் பூஜ்யம் பூஜ்யம் நான்கு ஒன்பது
              ⚠ தவறு: 500049 ஐ "ஐந்து லட்சத்து நாற்பத்தொன்பது" என்று சொல்லாதீர்கள்
  4. RERA — "/" segment:
              Rule A: எழுத்துக்கள் மட்டும் மற்றும் length > 5 → வார்த்தையாக
              Rule B: ≤5 chars அல்லது digit இருந்தால் → character-by-character (English)
  5. பரப்பு — 3-4 இலக்கங்கள் + "ச.அ."/"சதுர அடி"/"sq ft" → தமிழ் cardinal வார்த்தைகள்
              802 ச.அ. → எண்நூற்று இரண்டு சதுர அடி
  6. மற்றவை → தமிழ் cardinal வார்த்தைகள்

விலை (முழு உதாரணங்கள்):
  Rs. 44 லட்சம் → ரூபாய் நாற்பத்து நான்கு லட்சம்
  Rs. 85 லட்சம் → ரூபாய் எண்பத்தைந்து லட்சம் ← "ஐம்பது" அல்ல
  Rs. 64.99 லட்சம் → ரூபாய் அறுபத்து நான்கு புள்ளி ஒன்பது ஒன்பது லட்சம்

BHK:  2 BHK → இரண்டு BHK  |  3 BHK → மூன்று BHK  |  4 BHK → நான்கு BHK
""",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6B — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are Aarya, a warm and professional property sales advisor for Pacifica Companies.

━━━ RAG-ONLY POLICY ━━━
Answer EXCLUSIVELY from PROPERTY CONTEXT below.
• Answer IS in context → answer clearly and warmly.
• Answer NOT in context → say ONLY: "I'm sorry, I don't have that detail right now. Please reach out to our team directly!"
• NEVER invent prices, RERA numbers, areas, distances, or amenities.
• Do NOT use "typically", "usually", "in general".

━━━ PROPERTY NAME RULE ━━━
Every factual answer MUST mention the property name.
✓ "At Amara, the 2 BHK starts from Rs. 44 lakhs."
✓ "Aavaas Hyderabad is located at Bollaram Road, Miyapur."
✗ "The price starts from Rs. 44 lakhs." ← missing property name, WRONG

━━━ SINGLE vs MULTI PROPERTY ━━━
SINGLE (one === section ===): answer ONLY about that property, always name it.
COMPARISON (multiple === sections ===): list key facts per property clearly.

━━━ ONBOARDING ━━━
• __GREETING__: name unknown → introduce as Aarya, ask name. Name known → welcome back by name.
• When name given → thank, ask phone. Once both collected → never ask again.

━━━ RERA FORMAT ━━━
ui field: exact slash-format — PR/GJ/GANDHINAGAR/AUDA/RAA04324/A1R/211021

━━━ STYLE ━━━
Warm sentences. No bullets. 3–5 sentences max unless asked for more. End with a follow-up question.

━━━ OUTPUT — MANDATORY JSON ━━━
Two-step process:
  STEP 1 → Write "ui": factual answer in {language}, with raw numbers and symbols.
            For addresses/pincodes: keep as-is (e.g. Hyderabad-500049).
  STEP 2 → Write "tts": copy "ui" word-for-word, apply NUMBER TYPE DECISION TREE.
            CRITICAL: apply the correct rule for each number type — pincode ≠ price.

Output format (no markdown, no fences):
{{"ui": "<{language} — raw numbers, RERA slash-format>", "tts": "<{language} — numbers as spoken words per decision tree>"}}

{tts_rules}

LANGUAGE: Both fields entirely in {language_upper}.
"""

_LANG_INSTRUCTION = {
    "english":  "MANDATORY: Both fields in English only.",
    "hindi":    "अनिवार्य: दोनों fields पूरी तरह हिंदी में। पिनकोड को मौद्रिक राशि मत समझें।",
    "gujarati": "ફરજિયાત: બંને fields સંપૂર્ણ ગુજરાતીમાં. પિનકોડ ને ભાવ તરીકે ન ગણો — અંક-દ-અંક બોલો.",
    "telugu":   "తప్పనిసరి: రెండు fields తెలుగులో మాత్రమే. పిన్‌కోడ్‌ను ద్రవ్య మొత్తంగా చెప్పవద్దు.",
    "tamil":    "கட்டாயம்: இரண்டு fields தமிழில் மட்டுமே. பின்கோடை நிதி தொகையாக சொல்லாதீர்கள்.",
}

_LANG_STARTER = {
    "gujarati": "જી, ",
    "hindi":    "जी, ",
    "telugu":   "అవును, ",
    "tamil":    "ஆம், ",
    "english":  "",
}


def _build_messages(
    query: str, output_language: str, chat_history: Optional[List[dict]],
    context_chunks: List[str] = None, context_map: Optional[dict] = None,
    user_name: str = "", user_phone: str = "",
    intent: QueryIntent = "GENERAL",
) -> list:

    ONBOARD_KEYS = {"__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}
    is_onboard   = query.strip() in ONBOARD_KEYS

    if is_onboard or intent == "SKIP":
        context_str = "[ONBOARDING — NO PROPERTY CONTEXT NEEDED]"
    elif context_map:
        parts = []
        for col_id, chunks in context_map.items():
            if chunks:
                prop_name = col_id.replace("_col","").replace("_"," ").title()
                parts.append(f"=== {prop_name} ===\n" + "\n\n".join(chunks))
        context_str = "\n\n".join(parts) if parts else "[NO_CONTEXT_AVAILABLE]"
    elif context_chunks:
        context_str = "\n\n---\n\n".join(context_chunks)
    else:
        context_str = "[NO_CONTEXT_AVAILABLE]"

    lang      = output_language.lower()
    tts_rules = _TTS_NUMBER_RULES.get(lang, _TTS_NUMBER_RULES["english"])

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        language=lang,
        language_upper=lang.upper(),
        tts_rules=tts_rules,
    )

    user_ctx  = f"\nUser name: {user_name}" if user_name else "\nUser name: NOT YET COLLECTED"
    user_ctx += (f"\nUser phone: {user_phone} (collected — never ask again)"
                 if user_phone else "\nUser phone: NOT YET COLLECTED")

    system_content = f"{system_prompt}\n\n{user_ctx}\n\nPROPERTY CONTEXT:\n{context_str}"
    messages = [{"role": "system", "content": system_content}]

    history_limit = 4 if intent == "COMPARE" else 6
    if chat_history:
        for h in chat_history[-history_limit:]:
            if h["role"] == "assistant":
                try:
                    parsed = json.loads(h["content"])
                    messages.append({"role": "assistant",
                                     "content": parsed.get("ui", h["content"])})
                except Exception:
                    messages.append(h)
            else:
                messages.append(h)

    comparison_note = ""
    if intent == "COMPARE":
        comparison_note = (
            "[COMPARISON QUERY] Scan ALL === sections ===. "
            "For each property state its name and the relevant price/detail. "
            "Be concise — one line per property.\n\n"
        )

    lang_note = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["english"])
    messages.append({
        "role": "user",
        "content": (
            f"[{lang_note}]\n"
            f"[OUTPUT: Respond ONLY with JSON {{\"ui\": ..., \"tts\": ...}}]\n"
            f"[STEP 1: Write ui with exact numbers as-is. "
            f"STEP 2: Write tts — apply NUMBER TYPE DECISION TREE. "
            f"Pincode (6-digit in address) = digit-by-digit. "
            f"Price (before lakh/crore) = currency words. "
            f"Mobile (10-digit) = digit-by-digit. "
            f"Area sqft (3-4 digit + sqft) = cardinal words.]\n"
            f"[RERA in ui: slash format. In tts: per segment rules.]\n"
            f"{comparison_note}"
            f"{query}"
        )
    })

    starter = _LANG_STARTER.get(lang, "")
    if starter:
        messages.append({"role": "assistant",
                         "content": f'{{"ui": "{starter}'})
    return messages


def _parse_llm_json(raw: str) -> tuple[str, str]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
        return parsed.get("ui", text), parsed.get("tts", text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        return parsed.get("ui", text), parsed.get("tts", text)
    except json.JSONDecodeError:
        pass
    ui_m  = re.search(r'"ui"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    tts_m = re.search(r'"tts"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ui_m and tts_m:
        ui  = ui_m.group(1).replace('\\"', '"').replace('\\n', '\n')
        tts = tts_m.group(1).replace('\\"', '"').replace('\\n', '\n')
        return ui, tts
    log.warning("LLM JSON parse failed — using raw text for both ui and tts")
    return text, text


async def llm_answer(
    query: str, output_language: str = "english",
    chat_history: Optional[List[dict]] = None,
    context_chunks: List[str] = None,
    context_map: Optional[dict] = None,
    user_name: str = "", user_phone: str = "",
    intent: QueryIntent = "GENERAL",
) -> tuple[str, str]:
    messages = _build_messages(
        query, output_language, chat_history,
        context_chunks, context_map, user_name, user_phone, intent)

    ctx_chars = (
        sum(sum(len(c) for c in chunks) for chunks in context_map.values())
        if context_map else sum(len(c) for c in (context_chunks or []))
    )

    log.info("=" * 70)
    log.info("LLM INPUT — query: %s", query[:300])
    log.info("LLM INPUT — intent: %s | lang: %s | user: %s | phone_known: %s",
             intent, output_language, user_name or "[unknown]", bool(user_phone))
    if context_map:
        for cid, chunks in context_map.items():
            log.info("  ctx[%s]: %d chunks ~%d chars", cid, len(chunks), sum(len(c) for c in chunks))
    elif context_chunks:
        log.info("  ctx_chunks: %d ~%d chars", len(context_chunks), ctx_chars)
    else:
        log.info("  ctx: [NONE — SKIP/ONBOARD]")
    log.info("LLM INPUT — history turns: %d", len(chat_history or []))

    resp = await _openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw   = resp.choices[0].message.content.strip()
    p_tok = resp.usage.prompt_tokens
    c_tok = resp.usage.completion_tokens
    t_tok = resp.usage.total_tokens
    log.info("LLM OUTPUT — tokens: prompt=%d completion=%d total=%d  (~$%.4f @ gpt-4o)",
             p_tok, c_tok, t_tok, (p_tok * 2.5 + c_tok * 10) / 1_000_000)
    log.info("LLM RAW: %s", raw[:400])
    log.info("=" * 70)

    ui_text, tts_text = _parse_llm_json(raw)
    ui_text = normalise_rera(ui_text)

    log.info("UI : %s", ui_text[:300])
    log.info("TTS: %s", tts_text[:300])
    return ui_text, tts_text

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — TTS
# ─────────────────────────────────────────────────────────────────────────────

def _split_tts(text: str, limit: int = 490) -> List[str]:
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

def _tts_cleanup(text: str, language: str) -> str:
    text = re.sub(r'[*#~`\[\]{}]', '', text)
    text = re.sub(r'={2,}', '', text)
    text = re.sub(r'-{2,}', ', ', text)
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', ' and ', text)
    if language == "gujarati":
        text = re.sub(r"\([A-Za-z0-9\s,\.]+\)", "", text)
        text = re.sub(r"\s*।\s*", "। ", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text

_SARVAM_BASE = "https://api.sarvam.ai"
_http_client: Optional[httpx.AsyncClient] = None

def _get_http() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(40.0, connect=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
    return _http_client

async def sarvam_stt(audio_bytes: bytes, audio_format: str = "webm") -> str:
    log.info("STT REQUEST — format=%s size=%d bytes", audio_format, len(audio_bytes))
    headers = {"api-subscription-key": cfg.sarvam_api_key}
    files   = {"file": (f"audio.{audio_format}", audio_bytes, f"audio/{audio_format}")}
    data    = {"model": "saarika:v2.5", "language_code": "unknown", "with_diarization": "false"}
    async with httpx.AsyncClient(timeout=40) as client:
        resp = await client.post(f"{_SARVAM_BASE}/speech-to-text",
                                 headers=headers, files=files, data=data)
        resp.raise_for_status()
    result = resp.json()
    transcript    = result.get("transcript", "")
    lang_detected = result.get("language_code", "unknown")
    log.info("STT RESULT — lang_detected=%s transcript=%s", lang_detected, transcript[:300])
    return transcript

async def _tts_chunk(text: str, lang_code: str, speaker: str) -> bytes:
    headers = {"api-subscription-key": cfg.sarvam_api_key, "Content-Type": "application/json"}
    payload = {
        "inputs": [text], "target_language_code": lang_code, "speaker": speaker,
        "model": cfg.sarvam_tts_model, "enable_preprocessing": True,
        "speech_sample_rate": 22050, "enc_format": "wav", "loudness_normalization": True,
    }
    resp = await _get_http().post(f"{_SARVAM_BASE}/text-to-speech",
                                   headers=headers, json=payload)
    if resp.status_code != 200:
        log.error("Sarvam TTS %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
    audios = resp.json().get("audios", [])
    if not audios:
        raise ValueError("No audio from Sarvam TTS")
    return base64.b64decode(audios[0])

async def sarvam_tts(tts_text: str, language: str = "english") -> bytes:
    lang_code = _LANG_CODE.get(language.lower(), "en-IN")
    speaker   = _LANG_SPEAKER.get(language.lower(), "ritu")
    processed = _tts_cleanup(tts_text, language)
    log.info("TTS INPUT (original) : %s", tts_text[:400])
    log.info("TTS INPUT (processed): %s", processed[:400])
    chunks = _split_tts(processed)
    log.info("TTS REQUEST — %d chunk(s) | lang=%s | speaker=%s | model=%s",
             len(chunks), lang_code, speaker, cfg.sarvam_tts_model)
    for i, ch in enumerate(chunks):
        log.info("TTS CHUNK[%d]: %s", i, ch[:200])
    if len(chunks) == 1:
        wav = await _tts_chunk(chunks[0], lang_code, speaker)
        log.info("TTS RESPONSE — single chunk, wav size=%d bytes", len(wav))
        return wav
    wav_parts = await asyncio.gather(*[_tts_chunk(c, lang_code, speaker) for c in chunks])
    first           = wav_parts[0]
    sample_rate     = int.from_bytes(first[24:28], "little")
    num_channels    = int.from_bytes(first[22:24], "little")
    bits_per_sample = int.from_bytes(first[34:36], "little")
    pcm_data        = b"".join(w[44:] for w in wav_parts)
    data_size       = len(pcm_data)
    byte_rate       = sample_rate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    header = (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE" +
        b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                              byte_rate, block_align, bits_per_sample) +
        b"data" + struct.pack("<I", data_size))
    combined = header + pcm_data
    log.info("TTS RESPONSE — %d chunks merged, wav size=%d bytes", len(wav_parts), len(combined))
    return combined

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Startup — PropVoice v6.0.0 — initialising DB and ingesting documents …")
    Path(cfg.rag_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, rag_ingest_all)
    for r in results:
        log.info("  %-50s → %-10s (%d chunks)",
                 r.get("filename","?"), r.get("status","?"), r.get("chunks",0))
    yield
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    log.info("Shutdown complete.")

app = FastAPI(title="PropVoice — Pacifica Companies RAG API", version="6.0.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

_static_dir = Path("./static")
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


class UserInfo(BaseModel):
    session_id: str
    name:       str = ""
    phone:      str = ""

class ChatRequest(BaseModel):
    query:           str
    session_id:      str             = "default"
    pdf_id:          Optional[str]   = None
    pdf_ids:         Optional[List[str]] = None
    output_language: str             = "english"

class ChatResponse(BaseModel):
    answer_text:     str
    audio_base64:    Optional[str]   = None
    pdf_id:          Optional[str]   = None
    pdf_ids:         Optional[List[str]] = None
    output_language: str
    intent:          str             = "GENERAL"


async def _handle_query(
    query: str, session_id: str, output_language: str,
    pdf_id: str = "", pdf_ids_str: str = "",
    pdf_ids_list: Optional[List[str]] = None,
) -> ChatResponse:
    user    = db_get_user(session_id)
    u_name  = user.get("name", "")
    u_phone = user.get("phone", "")

    ONBOARD = {"__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}

    intent = "ONBOARD" if query.strip() in ONBOARD else classify_query(query)
    log.info("INTENT: %s | query: %s", intent, query[:100])

    target_ids: List[str] = []
    if pdf_ids_list:
        target_ids = pdf_ids_list
    elif pdf_ids_str.strip():
        target_ids = [x.strip() for x in pdf_ids_str.split(",") if x.strip()]
    elif pdf_id.strip():
        target_ids = [pdf_id.strip()]
    else:
        target_ids = [d["id"] for d in rag_list_docs()]

    loop = asyncio.get_event_loop()
    context_chunks: List[str]      = []
    context_map:    Optional[dict] = None

    if intent in ("SKIP", "ONBOARD"):
        pass
    elif len(target_ids) == 1:
        context_chunks = await loop.run_in_executor(
            None, rag_retrieve, target_ids[0], query, intent)
    elif len(target_ids) > 1:
        context_map = await loop.run_in_executor(
            None, lambda: rag_retrieve_multi(target_ids, query, intent))

    history = db_get_history(session_id, limit=10)
    db_save_turn(session_id, "user", query, output_language, ",".join(target_ids))

    ui_text, tts_text = await llm_answer(
        query, output_language, history,
        context_chunks, context_map, u_name, u_phone, intent)

    stored_content = json.dumps({"ui": ui_text, "tts": tts_text}, ensure_ascii=False)
    db_save_turn(session_id, "assistant", stored_content, output_language, ",".join(target_ids))

    audio_b64: Optional[str] = None
    try:
        wav       = await sarvam_tts(tts_text, output_language)
        audio_b64 = base64.b64encode(wav).decode()
        log.info("TTS SUCCESS — audio_b64 length=%d", len(audio_b64))
    except Exception as e:
        log.warning("TTS FAILED — %s", e)

    return ChatResponse(
        answer_text=ui_text,
        audio_base64=audio_b64,
        pdf_id=pdf_id or None,
        pdf_ids=target_ids if len(target_ids) > 1 else None,
        output_language=output_language,
        intent=intent,
    )


@app.get("/", include_in_schema=False)
async def serve_index():
    idx = _static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx), media_type="text/html")
    return {"message": "PropVoice API v6 running.", "docs": "/docs"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "6.0.0"}

@app.get("/api/pdfs")
async def get_pdfs():
    docs = rag_list_docs()
    return {"pdfs": docs, "count": len(docs)}

@app.post("/api/user", tags=["Users"])
async def save_user(info: UserInfo):
    db_upsert_user(info.session_id, info.name.strip(), info.phone.strip())
    return {"status": "saved", "session_id": info.session_id}

@app.get("/api/user/{session_id}", tags=["Users"])
async def get_user(session_id: str):
    user = db_get_user(session_id)
    return user if user else {"session_id": session_id, "name": "", "phone": ""}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "query cannot be empty")
    log.info("─" * 70)
    log.info("CHAT REQUEST — session=%s lang=%s query=%s",
             req.session_id, req.output_language, query[:200])
    return await _handle_query(
        query=query, session_id=req.session_id,
        output_language=req.output_language,
        pdf_ids_list=req.pdf_ids or ([req.pdf_id] if req.pdf_id else None))

@app.post("/api/voice", response_model=ChatResponse)
async def voice_endpoint(
    audio:           UploadFile = File(...),
    session_id:      str        = Form("default"),
    pdf_id:          str        = Form(""),
    pdf_ids:         str        = Form(""),
    output_language: str        = Form("english"),
):
    log.info("─" * 70)
    log.info("VOICE REQUEST — session=%s lang=%s", session_id, output_language)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Audio empty")
    try:
        transcript = await sarvam_stt(audio_bytes, "webm")
    except Exception as e:
        raise HTTPException(502, f"STT error: {e}")
    if not transcript.strip():
        raise HTTPException(422, "Could not transcribe — please speak clearly")
    log.info("VOICE TRANSCRIPT — %s", transcript[:300])
    return await _handle_query(
        query=transcript, session_id=session_id,
        output_language=output_language,
        pdf_id=pdf_id, pdf_ids_str=pdf_ids)

@app.get("/api/history/{session_id}", tags=["Users"])
async def get_history(session_id: str):
    raw_history = db_get_history(session_id, limit=50)
    clean = []
    for h in raw_history:
        if h["role"] == "assistant":
            try:
                parsed = json.loads(h["content"])
                clean.append({"role": "assistant", "content": parsed.get("ui", h["content"])})
            except Exception:
                clean.append(h)
        else:
            clean.append(h)
    return {"history": clean}

@app.post("/api/ingest", tags=["Admin"])
async def ingest_all(background_tasks: BackgroundTasks):
    background_tasks.add_task(rag_ingest_all)
    return {"message": "Ingestion started — will re-tag all chunks with schema_version=2"}

@app.delete("/api/cache", tags=["Admin"])
async def clear_cache():
    n = len(_embed_cache)
    _embed_cache.clear()
    return {"cleared": n}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=cfg.host, port=cfg.port, reload=True, log_level="info")