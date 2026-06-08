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
    rag_n_results:      int = Field(8,                env="RAG_N_RESULTS")

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
_rag_cache:   dict = {}
_embed_cache: dict = {}
_CACHE_MAX       = 512
_EMBED_CACHE_MAX = 256

import openai as _openai_sync
_openai_sync_client = _openai_sync.OpenAI(api_key=cfg.openai_api_key)

def _col_name(filename: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name + "_col")[:63]

def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks

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
        if col.metadata.get("file_hash") == current_hash and col.count() > 0:
            return {"status": "skipped", "collection": col_name, "chunks": col.count()}
        _chroma.delete_collection(col_name)
    except Exception:
        pass
    full_text = _extract_text(doc_path)
    if not full_text.strip():
        log.warning("No text extracted from %s", filename)
        col = _chroma.get_or_create_collection(name=col_name, embedding_function=_embed,
                  metadata={"file": filename, "file_hash": current_hash})
        return {"status": "empty", "collection": col_name, "chunks": 0}
    chunks = _chunk_text(full_text)
    col = _chroma.get_or_create_collection(name=col_name, embedding_function=_embed,
              metadata={"file": filename, "file_hash": current_hash})
    for start in range(0, len(chunks), 100):
        batch = chunks[start: start + 100]
        col.add(documents=batch,
                ids=[f"{col_name}_{start+i}" for i in range(len(batch))],
                metadatas=[{"source": filename, "chunk_index": start+i} for i in range(len(batch))])
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
    key = hashlib.sha256(query.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    resp   = _openai_sync_client.embeddings.create(input=[query], model="text-embedding-3-small")
    vector = resp.data[0].embedding
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        del _embed_cache[next(iter(_embed_cache))]
    _embed_cache[key] = vector
    return vector

def rag_retrieve(collection_id: str, query: str, n: int = None) -> List[str]:
    """Retrieve ALL chunks from a single-property collection — always give full context."""
    try:
        col = _chroma.get_collection(name=collection_id)
    except Exception:
        return []
    count = col.count()
    if count == 0:
        return []
    # For single-property queries, fetch ALL chunks (docs are small, 3-10 chunks).
    # This guarantees the LLM always has complete information.
    res    = col.get(include=["documents"])
    chunks = res.get("documents", [])
    log.info("RAG single-col %s: fetched ALL %d chunks", collection_id, len(chunks))
    return chunks

def _is_comparison_query(query: str) -> bool:
    lower = query.lower()
    patterns = [
        r"compar",
        r"\b(vs\.?|versus|difference|better|best|worst|cheapest|most\s+expensive|"
        r"lowest|highest|minimum|maximum|least|rank|which one|all sites?|all projects?|"
        r"all propert|across all|among all|between)\b",
        r"\b(price list|price range|cost of all|rates? of all|how much.*all|all.*price|"
        r"pricing.*all|all.*cost|starting price|starting cost)\b",
        r"\b(every project|each project|each property|all (the )?propert|full list|overview of all)\b",
        r"(સૌથી\s+(ઓછ|વધાર|મોંઘ|સસ્ત)|બધી\s+(સાઈટ|પ્રોજેક્ટ)|"
        r"કઈ\s+સાઈટ|સૌથી\s+ઓછો\s+ભાવ|સૌથી\s+વધ)",
        r"(सबसे\s+(कम|ज़्यादा|ज्यादा|महंगा|सस्ता)|"
        r"सभी\s+(साइट|प्रोजेक्ट|प्रॉपर्टी)|कौन\s+सी\s+साइट|तुलना)",
    ]
    return any(re.search(p, lower) for p in patterns)

def _extract_pricing_chunks(col_name: str) -> List[str]:
    PRICE_KW = re.compile(
        r"(price|cost|rate|lakh|crore|Rs\.|₹|starting|sqft|sq\.?\s*ft|"
        r"bhk|flat|unit|carpet|area|per\s+sq|onwards|pricing)", re.IGNORECASE)
    try:
        col   = _chroma.get_collection(name=col_name)
        count = col.count()
        if count == 0:
            return []
        res   = col.get(include=["documents"])
        docs  = res.get("documents", [])
        priced = [d for d in docs if PRICE_KW.search(d)]
        return priced if priced else docs
    except Exception as e:
        log.warning("Price chunk extract failed for %s: %s", col_name, e)
        return []

def rag_retrieve_multi(collection_ids: List[str], query: str) -> dict:
    is_comparison = _is_comparison_query(query)
    vector        = _get_query_embedding(query)

    if is_comparison:
        log.info("Comparison query — fetching PRICING chunks from every collection")

    def _query_one(cid: str) -> tuple:
        if is_comparison:
            return cid, _extract_pricing_chunks(cid)
        # Non-comparison multi-property: semantic search
        try:
            col   = _chroma.get_collection(name=cid)
            count = col.count()
            if count == 0:
                return cid, []
            n      = min(cfg.rag_n_results, count)
            res    = col.query(query_embeddings=[vector], n_results=n)
            chunks = res.get("documents", [[]])[0]
            return cid, chunks
        except Exception as e:
            log.warning("RAG retrieve failed for %s: %s", cid, e)
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
#  SECTION 5 — RERA NORMALISER  (LLM output cleanup)
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
#  SECTION 6 — SINGLE LLM CALL  (dual-output: ui_text + tts_text)
#
#  KEY ARCHITECTURE:
#  One call produces a JSON with two fields:
#    "ui"  — display text: raw numbers, RERA codes as-is, native script
#    "tts" — spoken text : all numbers as words in target language,
#            RERA digit-by-digit, currency as words
#
#  This eliminates ALL post-processing regex for numbers.
#  The LLM knows the language and writes both versions correctly in one pass.
# ─────────────────────────────────────────────────────────────────────────────

_openai = AsyncOpenAI(api_key=cfg.openai_api_key)

# ── TTS number/RERA instructions per language ────────────────────────────────
_TTS_NUMBER_RULES = {
    "english": """TTS NUMBER RULES (for the "tts" field — English):
- All integers and decimals → full English words. Up to 99 lakh: word form. Above: digit-by-digit.
  Examples: 44 → "forty four", 861 → "eight hundred sixty one", 61.33 → "sixty one point three three"
- Prices: Rs./₹ → "rupees", then number words + unit.
  Examples: "Rs. 44 lakhs" → "rupees forty four lakhs", "Rs. 1.20 Crores" → "rupees one point two zero crores"
- RERA codes — TWO rules for segments split by "/":
    • Segment is ONLY letters AND longer than 5 chars → speak as a natural word (city/district name).
      Examples: GANDHINAGAR → "Gandhinagar", AHMEDABAD → "Ahmedabad", SANAND → "Sanand"
    • Segment is 5 chars or shorter, OR contains any digit → spell every character individually.
      Examples: PR → "P R", GJ → "G J", AUDA → "A U D A", RAA04324 → "R A A zero four three two four",
                A1R → "A one R", 211021 → "two one one zero two one"
  Full example: PR/GJ/GANDHINAGAR/GANDHINAGAR/AUDA/RAA04324/A1R/211021
             → "P R, G J, Gandhinagar, Gandhinagar, A U D A, R A A zero four three two four, A one R, two one one zero two one"
  Full example: PR/GJ/AHMEDABAD/SANAND/AUDA/RAA07702/241120
             → "P R, G J, Ahmedabad, Sanand, A U D A, R A A zero seven seven zero two, two four one one two zero"
- Phone numbers: digit by digit. 9876543210 → "nine eight seven six five four three two one zero"
- BHK: digit as word. "2 BHK" → "two BHK", "3 BHK" → "three BHK"
- sq ft: "eight hundred sixty one square feet"
- Keep all other text natural and fluent.""",

    "hindi": """TTS संख्या नियम ("tts" फ़ील्ड के लिए — हिंदी):
- सभी संख्याएँ हिंदी शब्दों में लिखें।
  उदाहरण: 44 → "चौवालीस", 861 → "आठ सौ इकसठ", 2 → "दो", 3 → "तीन"
- कीमत: ₹/Rs. को "रुपये" लिखें, फिर हिंदी में संख्या + इकाई।
  उदाहरण: "Rs. 44 lakhs" → "रुपये चौवालीस लाख", "Rs. 61.33 lakhs" → "रुपये इकसठ दशमलव तैंतीस लाख"
- RERA कोड — "/" से अलग हर segment के लिए दो नियम:
    • Segment में ONLY letters हों AND 5 से ज़्यादा chars हों → पूरा शब्द बोलें (शहर/जिले का नाम)।
      उदाहरण: GANDHINAGAR → "Gandhinagar", AHMEDABAD → "Ahmedabad", SANAND → "Sanand"
    • Segment 5 या उससे कम chars का हो, OR कोई digit हो → हर character अलग-अलग बोलें।
      उदाहरण: PR → "P R", GJ → "G J", AUDA → "A U D A", RAA07702 → "R A A zero seven seven zero two",
               241120 → "two four one one two zero"
  पूरा उदाहरण: PR/GJ/AHMEDABAD/SANAND/AUDA/RAA07702/241120
            → "P R, G J, Ahmedabad, Sanand, A U D A, R A A zero seven seven zero two, two four one one two zero"
  (RERA के letters/numbers English में ही रखें)
- फ़ोन नंबर: अंक दर अंक।
- BHK: "2 BHK" → "दो BHK", "3 BHK" → "तीन BHK"
- वर्ग फुट: "आठ सौ इकसठ वर्ग फुट"
- बाकी सब text प्राकृतिक और धाराप्रवाह रखें।""",

    "gujarati": """TTS સંખ્યા નિયમો ("tts" ક્ષેત્ર માટે — ગુજરાતી):
- બધી સંખ્યાઓ ગુજરાતી શબ્દોમાં લખો।
  ઉદાહરણ: 44 → "ચુમ્માળીસ", 861 → "આઠ સો એકસઠ", 2 → "બે", 3 → "ત્રણ"
- કિંમત: ₹/Rs. → "રૂપિયા", પછી ગુજરાતીમાં સંખ્યા + એકમ।
  ઉદાહરણ: "Rs. 44 lakhs" → "રૂપિયા ચુમ્માળીસ લાખ", "Rs. 1.20 Crores" → "રૂપિયા એક કરોડ વીસ લાખ"
- RERA કોડ — "/" વડે અલગ થયેલ દરેક segment માટે બે નિયમ:
    • Segment માં ONLY letters હોય AND 5 થી વધુ chars હોય → આખો શબ્દ બોલો (શહેર/જિલ્લાનું નામ)।
      ઉદાહરણ: GANDHINAGAR → "Gandhinagar", AHMEDABAD → "Ahmedabad", SANAND → "Sanand"
    • Segment 5 અથવા તેથી ઓછા chars નો હોય, OR કોઈ digit હોય → દરેક character અલગ-અલગ બોલો।
      ઉદાહરણ: PR → "P R", GJ → "G J", AUDA → "A U D A", RAA04324 → "R A A zero four three two four",
               A1R → "A one R", 211021 → "two one one zero two one"
  પૂર્ણ ઉદાહરણ: PR/GJ/GANDHINAGAR/GANDHINAGAR/AUDA/RAA04324/A1R/211021
             → "P R, G J, Gandhinagar, Gandhinagar, A U D A, R A A zero four three two four, A one R, two one one zero two one"
  (RERA ના letters/numbers English char-by-char જ રાખો)
- ફોન નંબર: અંક-દર-અંક।
- BHK: "2 BHK" → "બે BHK", "3 BHK" → "ત્રણ BHK"
- ચોરસ ફૂટ: "આઠ સો એકસઠ ચોરસ ફૂટ"
- બાકી બધો text સ્વાભાવિક અને પ્રવાહી રાખો।""",

    "telugu": """TTS సంఖ్య నియమాలు ("tts" ఫీల్డ్ కోసం — తెలుగు):
- అన్ని సంఖ్యలు తెలుగు పదాలలో రాయండి.
  ఉదాహరణ: 44 → "నలభై నాలుగు", 861 → "ఎనిమిది వందల అరవై ఒకటి"
- ధర: ₹/Rs. → "రూపాయలు", తర్వాత తెలుగులో సంఖ్య + యూనిట్.
  ఉదాహరణ: "Rs. 44 lakhs" → "రూపాయలు నలభై నాలుగు లక్షలు"
- RERA కోడ్ — "/" తో విభజించిన ప్రతి segment కు రెండు నియమాలు:
    • Segment లో ONLY letters ఉంటే AND 5 కంటే ఎక్కువ chars ఉంటే → మొత్తం పదంగా చదవండి (నగరం/జిల్లా పేరు).
      ఉదాహరణ: GANDHINAGAR → "Gandhinagar", AHMEDABAD → "Ahmedabad", HYDERABAD → "Hyderabad"
    • Segment 5 లేదా తక్కువ chars, లేదా digit ఉంటే → ప్రతి character విడివిడిగా చదవండి.
      ఉదాహరణ: PR → "P R", GJ → "G J", AUDA → "A U D A", RAA07702 → "R A A zero seven seven zero two"
  (RERA letters/numbers English లో char-by-char మాట్లాడండి)
- ఫోన్ నంబర్: అంకె వారీగా.
- BHK: "2 BHK" → "రెండు BHK"
- మిగతా text సహజంగా ఉంచండి.""",

    "tamil": """TTS எண் விதிகள் ("tts" புலத்திற்கு — தமிழ்):
- அனைத்து எண்களையும் தமிழ் வார்த்தைகளில் எழுதவும்.
  எடுத்துக்காட்டு: 44 → "நாற்பத்து நான்கு", 861 → "எட்டு நூற்று அறுபத்து ஒன்று"
- விலை: ₹/Rs. → "ரூபாய்", பின்னர் தமிழில் எண் + அலகு.
  எடுத்துக்காட்டு: "Rs. 44 lakhs" → "ரூபாய் நாற்பத்து நான்கு லட்சம்"
- RERA குறியீடு — "/" மூலம் பிரிக்கப்பட்ட ஒவ்வொரு segment க்கும் இரண்டு விதிகள்:
    • Segment ல் ONLY letters இருந்தால் AND 5 எழுத்துகளுக்கு மேல் இருந்தால் → முழு வார்த்தையாக படிக்கவும் (நகர/மாவட்டப் பெயர்).
      எடுத்துக்காட்டு: GANDHINAGAR → "Gandhinagar", AHMEDABAD → "Ahmedabad", CHENNAI → "Chennai"
    • Segment 5 அல்லது குறைவான எழுத்துகள், அல்லது digit இருந்தால் → ஒவ்வொரு character தனியாக படிக்கவும்.
      எடுத்துக்காட்டு: PR → "P R", GJ → "G J", AUDA → "A U D A", RAA04324 → "R A A zero four three two four"
  (RERA letters/numbers ஆங்கிலத்தில் char-by-char பேசவும்)
- தொலைபேசி எண்: இலக்கம் இலக்கமாக.
- BHK: "2 BHK" → "இரண்டு BHK"
- மற்ற உரையை இயற்கையாக வைத்திருங்கள்.""",
}

_SYSTEM_PROMPT_TEMPLATE = """You are Aarya, a warm property sales advisor for Pacifica Companies.

════════════════════════════════════════
STRICT RAG-ONLY POLICY
════════════════════════════════════════
Answer EXCLUSIVELY from the PROPERTY CONTEXT below.

Rules:
1. If the answer IS in PROPERTY CONTEXT → answer clearly and warmly.
2. If the answer is NOT in context → say only: "I'm sorry, I don't have that detail right now. Please reach out to our team directly!"
   NEVER say "for all projects" or generalise — you are focused on the selected property/properties.
3. NEVER invent prices, RERA numbers, distances, amenities, or any fact.
4. Do NOT use phrases like "typically", "usually", "in general".

SINGLE PROPERTY RULE:
When context has only one === section ===, answer ONLY about that property.
Never say "I don't have details for all projects" — you only have ONE project. Just answer it.

COMPARISON RULE:
When context has multiple === sections ===:
- Scan all sections for the requested data point.
- State ONLY the winner (lowest/highest). Do not list all properties.
- Example: "The lowest 2 BHK price is at Madrid County starting from Rs. 41.14 lakhs."

ONBOARDING:
- __GREETING__: If name unknown → introduce as Aarya, ask name. If name known → welcome back by name.
- When name given → thank them, ask phone number.
- Once name+phone collected → never ask again.

RERA FORMAT (ui field):
Write RERA exactly as in source: PR/GJ/GANDHINAGAR/AUDA/RAA04324/A1R/211021
Never break it into P R, G J... in the ui field.

STYLE:
- Warm, natural sentences. No bullet points.
- 3–5 sentences max unless user asks for more.
- End with a gentle follow-up question when appropriate.

════════════════════════════════════════
OUTPUT FORMAT — MANDATORY JSON
════════════════════════════════════════
You MUST respond with ONLY a valid JSON object — no markdown, no extra text, no ```json fences.
Format:
{{
  "ui": "<response in {language} with raw numbers, Rs. notation, RERA codes as-is>",
  "tts": "<same response in {language} with ALL numbers as words — see TTS RULES below>"
}}

The "ui" field is displayed to the user exactly as written.
The "tts" field is sent to speech synthesis — every number must be spelled out as words.
Both fields must carry the SAME information and the SAME language.

{tts_rules}

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
Write BOTH "ui" and "tts" fields entirely in: {language_upper}
Exception: RERA codes in "ui" stay in slash format. In "tts" they are spoken char-by-char."""

_LANG_INSTRUCTION = {
    "english":  "MANDATORY: Both fields in English only.",
    "hindi":    "अनिवार्य: दोनों fields पूरी तरह हिंदी में।",
    "gujarati": "ફરજિયાત: બંને fields સંપૂર્ણ ગુજરાતીમાં.",
    "telugu":   "తప్పనిసరి: రెండు fields తెలుగులో మాత్రమే.",
    "tamil":    "கட்டாயம்: இரண்டு fields தமிழில் மட்டுமே.",
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
) -> list:

    is_onboard = query.strip() in {
        "__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}
    is_comparison = _is_comparison_query(query)

    if is_onboard:
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

    lang     = output_language.lower()
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

    if chat_history:
        # Convert stored history: if assistant turns have JSON, show only ui text
        for h in chat_history[-8:]:
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
    if is_comparison:
        comparison_note = (
            "[COMPARISON QUERY] Scan ALL === sections ===. "
            "Answer ONLY with the winner property and its price. "
            "Do NOT list all properties.\n\n"
        )

    lang_note = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["english"])
    messages.append({
        "role": "user",
        "content": (
            f"[{lang_note}]\n"
            f"[OUTPUT: Respond ONLY with a JSON object {{\"ui\": ..., \"tts\": ...}}]\n"
            f"[RERA in ui: slash format only — PR/GJ/... Never break it]\n"
            f"{comparison_note}"
            f"{query}"
        )
    })

    starter = _LANG_STARTER.get(lang, "")
    if starter:
        # Prime the assistant to start with the correct language
        messages.append({"role": "assistant",
                         "content": f'{{"ui": "{starter}'})

    return messages


def _parse_llm_json(raw: str) -> tuple[str, str]:
    """
    Parse the dual-output JSON from LLM.
    Returns (ui_text, tts_text).
    Falls back gracefully if JSON is malformed.
    """
    # If we primed the assistant, the raw starts from after the opening brace
    # The full JSON may be: {"ui": "...", "tts": "..."} or just the continuation
    text = raw.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        return parsed.get("ui", text), parsed.get("tts", text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        return parsed.get("ui", text), parsed.get("tts", text)
    except json.JSONDecodeError:
        pass

    # Try extracting ui/tts with regex
    ui_m  = re.search(r'"ui"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    tts_m = re.search(r'"tts"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if ui_m and tts_m:
        ui  = ui_m.group(1).replace('\\"', '"').replace('\\n', '\n')
        tts = tts_m.group(1).replace('\\"', '"').replace('\\n', '\n')
        return ui, tts

    # Last resort: return raw as both
    log.warning("LLM JSON parse failed — using raw text for both ui and tts")
    return text, text


async def llm_answer(
    query: str, output_language: str = "english",
    chat_history: Optional[List[dict]] = None,
    context_chunks: List[str] = None,
    context_map: Optional[dict] = None,
    user_name: str = "", user_phone: str = "",
) -> tuple[str, str]:
    """
    Single LLM call.
    Returns (ui_text, tts_text).
    """
    messages = _build_messages(
        query, output_language, chat_history,
        context_chunks, context_map, user_name, user_phone)

    log.info("=" * 70)
    log.info("LLM INPUT — query: %s", query[:300])
    log.info("LLM INPUT — language: %s | user: %s | phone_known: %s",
             output_language, user_name or "[unknown]", bool(user_phone))
    if context_map:
        for cid, chunks in context_map.items():
            log.info("LLM INPUT — context[%s]: %d chunks, ~%d chars",
                     cid, len(chunks), sum(len(c) for c in chunks))
    elif context_chunks:
        log.info("LLM INPUT — context_chunks: %d, ~%d chars",
                 len(context_chunks), sum(len(c) for c in context_chunks))
    else:
        log.info("LLM INPUT — context: [NONE]")
    log.info("LLM INPUT — history turns: %d", len(chat_history or []))

    resp = await _openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=800,
        temperature=0.1,
        response_format={"type": "json_object"},   # force JSON mode
    )
    raw = resp.choices[0].message.content.strip()
    log.info("LLM OUTPUT (raw): %s", raw[:500])
    log.info("LLM OUTPUT — tokens: prompt=%d completion=%d total=%d",
             resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens)
    log.info("=" * 70)

    ui_text, tts_text = _parse_llm_json(raw)

    # Normalise RERA in ui_text only (tts_text already has spoken form from LLM)
    ui_text = normalise_rera(ui_text)

    log.info("UI  TEXT: %s", ui_text[:300])
    log.info("TTS TEXT: %s", tts_text[:300])

    return ui_text, tts_text

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — TTS  (minimal post-processing — LLM handles numbers)
# ─────────────────────────────────────────────────────────────────────────────

def _split_tts(text: str, limit: int = 490) -> List[str]:
    """Split on sentence boundaries, never exceed limit chars (Sarvam max=500)."""
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
    """
    Light cleanup only — LLM already produced spoken-form numbers in tts field.
    We only strip markdown symbols and fix Gujarati punctuation.
    """
    # Strip markdown
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
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
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
    log.info("Startup — initialising DB and ingesting documents …")
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

app = FastAPI(title="PropVoice — Pacifica Companies RAG API", version="4.0.0",
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


async def _handle_query(
    query: str, session_id: str, output_language: str,
    pdf_id: str = "", pdf_ids_str: str = "",
    pdf_ids_list: Optional[List[str]] = None,
) -> ChatResponse:
    """
    Core handler shared by /api/chat and /api/voice.
    Single LLM call → (ui_text, tts_text) → TTS → ChatResponse.
    """
    user    = db_get_user(session_id)
    u_name  = user.get("name", "")
    u_phone = user.get("phone", "")

    ONBOARD = {"__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}

    # Build target collection list
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
    context_chunks: List[str]     = []
    context_map:    Optional[dict] = None

    if query in ONBOARD:
        pass  # no RAG for onboarding
    elif len(target_ids) == 1:
        # Single property: fetch ALL chunks for complete context
        context_chunks = await loop.run_in_executor(
            None, rag_retrieve, target_ids[0], query)
    elif len(target_ids) > 1:
        context_map = await loop.run_in_executor(
            None, lambda: rag_retrieve_multi(target_ids, query))

    history = db_get_history(session_id, limit=10)
    db_save_turn(session_id, "user", query, output_language, ",".join(target_ids))

    # ONE LLM CALL — returns (ui_text, tts_text)
    ui_text, tts_text = await llm_answer(
        query, output_language, history,
        context_chunks, context_map, u_name, u_phone)

    # Store the JSON so history reconstruction works
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
    )


@app.get("/", include_in_schema=False)
async def serve_index():
    idx = _static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx), media_type="text/html")
    return {"message": "PropVoice API v4 running.", "docs": "/docs"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}

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
    # Return ui text only (not raw JSON)
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
    return {"message": "Ingestion started"}

@app.delete("/api/cache", tags=["Admin"])
async def clear_cache():
    n = len(_rag_cache)
    _rag_cache.clear()
    return {"cleared": n}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=cfg.host, port=cfg.port, reload=True, log_level="info")