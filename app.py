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
#  SECTION 2 — DETERMINISTIC NUM2WORDS ENGINE
#
#  Replaces the old NUMBER_MAP / dual-output LLM approach entirely.
#  The LLM now outputs ONE field (ui text with raw numbers).
#  This engine post-processes that text before TTS:
#    • Mobile numbers / pincodes / RERA codes  → digit-by-digit
#    • Amounts (₹ / Rs. / lakhs / crores)      → words via JSON dict
#    • Sq ft / sq. ft                          → words via JSON dict
#    • Bare integers                            → words via JSON dict
# ─────────────────────────────────────────────────────────────────────────────

# JSON files must be in ./lang_dicts/ directory.
# Expected filenames: gujarati_numbers.json, hindi_numbers.json,
#                     tamil_numbers.json, telugu_numbers.json, english_numbers.json
# Each file: {"0": "word", "1": "word", ..., up to at least 999}

_LANG_DICTS: dict[str, dict[int, str]] = {}

def _load_lang_dict(lang: str) -> dict[int, str]:
    if lang in _LANG_DICTS:
        return _LANG_DICTS[lang]
    path = Path(f"./lang_dicts/{lang}_numbers.json")
    if not path.exists():
        log.warning("Number dict not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    d = {int(k): v for k, v in raw.items()}
    _LANG_DICTS[lang] = d
    log.info("Loaded %d entries from %s", len(d), path)
    return d


# ── Per-language helpers ──────────────────────────────────────────────────────

_HUNDREDS = {
    "gujarati": ("એકસો", "સો"),
    "hindi":    ("एक सौ", "सौ"),
    "tamil":    ("நூறு",  "நூறு"),   # tamil uses நூறு for 100, NN நூறு for 200+
    "telugu":   ("వంద",  "వందలు"),  # approximate; JSON should cover up to 99
    "english":  ("one hundred", " hundred"),
}

def _below_1000(n: int, lang: str, d: dict[int, str]) -> str:
    """Convert 1–999 to words using the language dict."""
    if n == 0:
        return ""
    if n in d:
        return d[n]
    if n < 100:
        # fallback: shouldn't happen if JSON is complete 1-99
        return str(n)
    hundreds_digit = n // 100
    remainder      = n % 100
    one_h, multi_h = _HUNDREDS.get(lang, ("one hundred", " hundred"))
    if hundreds_digit == 1:
        h_word = one_h
    else:
        h_word = d.get(hundreds_digit, str(hundreds_digit)) + multi_h
    if remainder:
        return h_word + " " + (d.get(remainder, str(remainder)))
    return h_word


def _num_to_words(n: int, lang: str) -> str:
    """
    Convert any non-negative integer to words in the given language.
    Uses crore / lakh / thousand system (Indian numbering).
    """
    d = _load_lang_dict(lang)
    if not d:
        return str(n)
    if n == 0:
        return d.get(0, "zero")

    lang_units = {
        "gujarati": ("કરોડ",  "લાખ",  "હજાર"),
        "hindi":    ("करोड़", "लाख",  "हज़ार"),
        "tamil":    ("கோடி",  "லட்சம்", "ஆயிரம்"),
        "telugu":   ("కోటి",  "లక్ష",  "వేయి"),
        "english":  ("crore", "lakh",  "thousand"),
    }
    crore_w, lakh_w, thousand_w = lang_units.get(lang, lang_units["english"])

    parts   = []
    crore   = n // 10_000_000;  n %= 10_000_000
    lakh    = n // 100_000;     n %= 100_000
    thousand= n // 1_000;       n %= 1_000
    rem     = n

    if crore:
        parts.append(_below_1000(crore, lang, d) + " " + crore_w)
    if lakh:
        parts.append(_below_1000(lakh,  lang, d) + " " + lakh_w)
    if thousand:
        parts.append(_below_1000(thousand, lang, d) + " " + thousand_w)
    if rem:
        parts.append(_below_1000(rem, lang, d))

    return " ".join(parts)


# ── Digit-by-digit for RERA / mobile / pincode ───────────────────────────────

_DIGIT_WORDS = {
    "gujarati": ["શૂન્ય","એક","બે","ત્રણ","ચાર","પાંચ","છ","સાત","આઠ","નવ"],
    "hindi":    ["शून्य","एक","दो","तीन","चार","पाँच","छह","सात","आठ","नौ"],
    "tamil":    ["சுழியம்","ஒன்று","இரண்டு","மூன்று","நான்கு","ஐந்து","ஆறு","ஏழு","எட்டு","ஒன்பது"],
    "telugu":   ["సున్న","ఒకటి","రెండు","మూడు","నాలుగు","అయిదు","ఆరు","ఏడు","ఎనిమిది","తొమ్మిది"],
    "english":  ["zero","one","two","three","four","five","six","seven","eight","nine"],
}

def _digit_by_digit(token: str, lang: str) -> str:
    """Speak each character individually: digits as words, letters as-is."""
    dw   = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["english"])
    out  = []
    for ch in token:
        if ch.isdigit():
            out.append(dw[int(ch)])
        elif ch.isalpha():
            out.append(ch.upper())
        # skip punctuation like / - inside RERA
    return " ".join(out)


# ── RERA detection ────────────────────────────────────────────────────────────
# Matches patterns like PR/GJ/GANDHINAGAR/AUDA/RAA04324/A1R/211021

_RERA_PAT = re.compile(
    r'\b([A-Z]{1,3}/[A-Z]{2}/[A-Z0-9/]{4,})\b'
)

def _speak_rera(rera: str, lang: str) -> str:
    """
    Per-segment rules:
      - Segment is ONLY letters AND len > 5  → speak as a word (city/district name)
      - Segment is ≤5 chars OR contains digit → spell every character

    Segments are joined with ", " (comma+space) so the TTS engine inserts a
    natural pause between each segment — preventing the rushed run-together
    speech observed with tokens like SANAND.
    """
    segments = rera.split("/")
    spoken   = []
    for seg in segments:
        if seg.isalpha() and len(seg) > 5:
            spoken.append(seg.capitalize())   # e.g. GANDHINAGAR → Gandhinagar
        else:
            spoken.append(_digit_by_digit(seg, lang))
    # Comma between segments = TTS pause; keeps letters/words naturally separated
    return ", ".join(spoken)


# ── Mobile / phone detection (10-digit starting with 6-9, or with +91) ───────

_MOBILE_PAT = re.compile(r'(?<!\d)(\+91[-\s]?)?([6-9]\d{9})(?!\d)')

# ── Pincode detection (exactly 6 digits, Indian) ─────────────────────────────

_PINCODE_PAT = re.compile(r'(?<!\d)([1-9]\d{5})(?!\d)')

# ── Amount patterns ───────────────────────────────────────────────────────────
# Matches: Rs. 44.5 lakhs / ₹44,50,000 / 44.5 lakhs / 2.5 crores etc.

# ── Amount patterns — ADD Gujarati/Hindi/Tamil/Telugu unit words ──────────────

_AMOUNT_PAT = re.compile(
    r'(?:Rs\.?\s*|₹\s*|રૂ\.?\s*|रु\.?\s*)?'
    r'(\d[\d,]*(?:\.\d+)?)'
    r'\s*(lakh(?:s)?|crore(?:s)?|cr\.?|L\.?'
    r'|લાખ|કરોડ|लाख|करोड़?|லட்சம்|கோடி|లక్ష|కోటి)'
    , re.IGNORECASE
)


# ── Sq ft pattern ─────────────────────────────────────────────────────────────

_SQFT_PAT = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s*feet|sqft)',
    re.IGNORECASE
)

# ── Bare number (fallback for remaining standalone integers) ──────────────────

_BARE_NUM_PAT = re.compile(r'(?<![/\w])(\d{1,7})(?![/\w\d])')


_CURRENCY_PREFIX = {
    "gujarati": "રૂપિયા",
    "hindi":    "रुपये",
    "tamil":    "ரூபாய்",
    "telugu":   "రూపాయలు",
    "english":  "rupees",
}

_LAKH_WORD = {
    "gujarati": "લાખ",
    "hindi":    "लाख",
    "tamil":    "லட்சம்",
    "telugu":   "లక్ష",
    "english":  "lakh",
}

_CRORE_WORD = {
    "gujarati": "કરોડ",
    "hindi":    "करोड़",
    "tamil":    "கோடி",
    "telugu":   "కోటి",
    "english":  "crore",
}

_SQFT_WORD = {
    "gujarati": "સ્ક્વેર ફૂટ",
    "hindi":    "स्क्वेयर फ़ीट",
    "tamil":    "சதுர அடி",
    "telugu":   "చదరపు అడుగులు",
    "english":  "square feet",
}


def _parse_amount(num_str: str) -> float:
    """Parse '44,50,000' or '44.5' → float."""
    return float(num_str.replace(",", ""))


def _decimal_digits_to_words(dec_str: str, lang: str) -> str:
    """
    Speak each decimal digit individually as a word.
    e.g. ".33" → "तीन तीन" (NOT "तैंतीस")
    e.g. ".5"  → "पाँच"
    This avoids the bug where int("33") → 33 → thirty-three instead of three three.
    """
    dw = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["english"])
    return " ".join(dw[int(ch)] for ch in dec_str if ch.isdigit())

def _amount_to_words(num_str: str, unit: str, lang: str, has_currency_prefix: bool) -> str:
    """Convert '44.5 lakhs' or '61.33 lakhs' or '61.33 લાખ' → language words."""
    val  = _parse_amount(num_str)

    # Normalise native-script unit names → English routing keys
    _UNIT_NORM = {
        "લાખ": "lakh",  "કરોડ": "crore",
        "लाख": "lakh",  "करोड़": "crore", "करोड": "crore",
        "லட்சம்": "lakh", "கோடி": "crore",
        "లక్ష": "lakh", "కోటి": "crore",
    }
    unit_key = _UNIT_NORM.get(unit, unit.lower().rstrip("s").rstrip("."))

    currency = _CURRENCY_PREFIX.get(lang, "rupees")

    _POINT = {"gujarati":"પોઈન્ટ","hindi":"दशमलव","tamil":"புள்ளி",
              "telugu":"దశాంశం","english":"point"}

    def _with_decimal(whole_val: float, unit_w: str) -> str:
        whole   = int(whole_val)
        if "." in num_str:
            dec_str = num_str.split(".")[1].rstrip("0")
        else:
            dec_str = ""
        w_whole    = _num_to_words(whole, lang)
        point_word = _POINT.get(lang, "point")
        if dec_str:
            dec_words = _decimal_digits_to_words(dec_str, lang)
            if has_currency_prefix:
                return f"{currency} {w_whole} {point_word} {dec_words} {unit_w}"
            return f"{w_whole} {point_word} {dec_words} {unit_w}"
        else:
            if has_currency_prefix:
                return f"{currency} {w_whole} {unit_w}"
            return f"{w_whole} {unit_w}"

    if unit_key in ("lakh", "l"):
        return _with_decimal(val, _LAKH_WORD.get(lang, "lakh"))
    elif unit_key in ("crore", "cr"):
        return _with_decimal(val, _CRORE_WORD.get(lang, "crore"))

    return _num_to_words(int(val), lang)


def _sqft_to_words(num_str: str, lang: str) -> str:
    val    = _parse_amount(num_str)
    whole  = int(val)
    w      = _num_to_words(whole, lang)
    unit_w = _SQFT_WORD.get(lang, "square feet")
    return f"{w} {unit_w}"


def num2words_tts(ui_text: str, lang: str) -> str:
    """
    Post-process LLM ui_text for TTS:
      1. RERA codes          → per-segment rule (word or char-by-char)
      2. Mobile / phone      → digit-by-digit
      3. Pincode             → digit-by-digit
      4. Amount + unit       → word conversion (lakh/crore)
      5. Sq ft               → word conversion
      6. Bare standalone int → word conversion
    Order matters — process most-specific patterns first to avoid double-conversion.
    """
    lang = lang.lower()
    text = ui_text

    # ── 1. RERA ──────────────────────────────────────────────────────────────
    def _sub_rera(m: re.Match) -> str:
        return _speak_rera(m.group(1), lang)
    text = _RERA_PAT.sub(_sub_rera, text)

    # ── 2. Mobile numbers ────────────────────────────────────────────────────
    def _sub_mobile(m: re.Match) -> str:
        prefix = m.group(1) or ""
        number = m.group(2)
        spoken = _digit_by_digit(number, lang)
        if prefix.strip():
            # +91 → speak "plus" then 9 and 1 in native language
            dw        = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["english"])
            plus_word = {"gujarati":"પ્લસ","hindi":"प्लस","tamil":"பிளஸ்",
                         "telugu":"ప్లస్","english":"plus"}.get(lang, "plus")
            nine      = dw[9]
            one       = dw[1]
            return f"{plus_word} {nine} {one} {spoken}"
        return spoken
    text = _MOBILE_PAT.sub(_sub_mobile, text)

    # ── 3. Pincode ───────────────────────────────────────────────────────────
    # Only match 6-digit groups that look like pincodes (after mobile already removed)
    def _sub_pincode(m: re.Match) -> str:
        return _digit_by_digit(m.group(1), lang)
    text = _PINCODE_PAT.sub(_sub_pincode, text)

    # ── 4. Amount + unit ─────────────────────────────────────────────────────
    _AMT_FULL = re.compile(
        r'(Rs\.?\s*|₹\s*|રૂ\.?\s*|रु\.?\s*)?'
        r'(\d[\d,]*(?:\.\d+)?)'
        r'\s*(lakh(?:s)?|crore(?:s)?|cr\.?|L\.?'
        r'|લાખ|કરોડ|लाख|करोड़?|லட்சம்|கோடி|లక్ష|కోటి)',
        re.IGNORECASE
    )
    def _sub_amount(m: re.Match) -> str:
        has_prefix = bool(m.group(1) and m.group(1).strip())
        return _amount_to_words(m.group(2), m.group(3), lang, has_prefix)
    text = _AMT_FULL.sub(_sub_amount, text)

    # ── 5. Sq ft ─────────────────────────────────────────────────────────────
    def _sub_sqft(m: re.Match) -> str:
        return _sqft_to_words(m.group(1), lang)
    text = _SQFT_PAT.sub(_sub_sqft, text)

    # ── 6. Bare integers (1–7 digits, not inside words / already converted) ──
    def _sub_bare(m: re.Match) -> str:
        n = int(m.group(1))
        return _num_to_words(n, lang)
    text = _BARE_NUM_PAT.sub(_sub_bare, text)

    log.info("NUM2WORDS [%s] IN : %s", lang, ui_text[:300])
    log.info("NUM2WORDS [%s] OUT: %s", lang,    text[:300])
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — SQLITE CONVERSATION STORE
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
#  SECTION 4 — DOCX EXTRACTOR
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
#  SECTION 5 — CHROMADB + RAG
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

# ── Query synonym/expansion map ───────────────────────────────────────────────
# When a user's query matches a known intent, we embed an enriched version that
# covers all the ways that information might appear in property documents.
# This fixes cases like "mobile number" not finding "contact number" chunks.

_QUERY_SYNONYMS: List[tuple[re.Pattern, str]] = [
    # Contact / phone
    (re.compile(
        r'\b(mobile|phone|contact|helpline|call|reach|number|whatsapp|toll.?free)\b',
        re.IGNORECASE),
     "contact number phone mobile helpline call us reach"),

    # Price / cost
    (re.compile(
        r'\b(price|cost|rate|budget|how much|kitna|bhav|kimat|dam|mol)\b',
        re.IGNORECASE),
     "price cost rate starting price budget lakh crore BHK flat unit"),

    # Location / address
    (re.compile(
        r'\b(location|address|where|kahan|jagah|site|area|near|locality)\b',
        re.IGNORECASE),
     "location address site area nearby landmark distance"),

    # RERA
    (re.compile(r'\brera\b', re.IGNORECASE),
     "RERA registration number approval certificate"),

    # Amenities / facilities
    (re.compile(
        r'\b(amenity|amenities|facilities|features|club|gym|pool|park|garden|parking)\b',
        re.IGNORECASE),
     "amenities facilities clubhouse gym swimming pool garden parking"),

    # Possession / handover
    (re.compile(
        r'\b(possession|handover|ready|delivery|when|completion|kab)\b',
        re.IGNORECASE),
     "possession date ready to move handover completion"),

    # Size / carpet area
    (re.compile(
        r'\b(size|area|carpet|sqft|sq\.?\s*ft|square|feet|foot|dimensions)\b',
        re.IGNORECASE),
     "carpet area size sqft square feet dimensions flat size"),

    # BHK type
    (re.compile(r'\b(\d\s*bhk|bedroom|configuration|type)\b', re.IGNORECASE),
     "BHK bedroom configuration flat type 2BHK 3BHK 4BHK"),

    # Developer / builder
    (re.compile(
        r'\b(developer|builder|company|pacifica|promoter|who built)\b',
        re.IGNORECASE),
     "developer builder Pacifica Companies promoter"),
]

def _expand_query(query: str) -> str:
    """
    Return an enriched query string for embedding.
    If the query matches any known intent pattern, append relevant synonyms
    so the semantic search retrieves chunks that use different terminology.
    Original query words are always preserved.
    """
    lower = query.lower().strip()
    expansions = []
    for pattern, extra in _QUERY_SYNONYMS:
        if pattern.search(lower):
            expansions.append(extra)
    if expansions:
        expanded = query + " " + " ".join(expansions)
        log.info("QUERY EXPANDED: [%s] → [%s]", query[:80], expanded[:150])
        return expanded
    return query


def _get_query_embedding(query: str) -> list:
    # Expand query before embedding for better semantic recall
    expanded = _expand_query(query)
    key = hashlib.sha256(expanded.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    resp   = _openai_sync_client.embeddings.create(input=[expanded], model="text-embedding-3-small")
    vector = resp.data[0].embedding
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        del _embed_cache[next(iter(_embed_cache))]
    _embed_cache[key] = vector
    return vector

# ── Generic / company-level query detection ──────────────────────────────────
# These queries are about Pacifica the company, not any specific property.
# When no pdf_id is selected, route them to the generic info doc only.

_GENERIC_COMPANY_PAT = re.compile(
    r'\b(pacifica\s+companies?|pacifica\s+group|pacifica\s+realt|'
    r'who\s+(are|is)\s+(pacifica|you|the\s+company|the\s+developer)|'
    r'about\s+(pacifica|the\s+company|you)|'
    r'(company|developer|builder|promoter|group)\s*(info|detail|background|history|profile)|'
    r'(founded|established|hq|headquarter|origin|since|years?\s+old))\b',
    re.IGNORECASE
)

# Collection name fragment that maps to the generic Pacifica info doc
_GENERIC_COL_FRAGMENT = "Pacifica_Companies"


def _find_generic_col(all_ids: List[str]) -> Optional[str]:
    """Return the collection ID for the generic Pacifica info doc, if present."""
    for cid in all_ids:
        if _GENERIC_COL_FRAGMENT in cid:
            return cid
    return None


# ── Property name → collection routing ───────────────────────────────────────
# Maps common property name mentions in queries to their collection IDs.
# Built dynamically from rag_list_docs() at first use.

_PROP_NAME_MAP: dict[str, str] = {}   # filled lazily in _resolve_target_ids()

def _build_prop_name_map(all_docs: List[dict]) -> dict[str, str]:
    """
    Build {keyword → collection_id} from document display names.
    e.g. "amara" → "Amara_and_North_Enclave_col"
         "north enclave" → "Amara_and_North_Enclave_col"
         "madrid" → "Madrid_County_col"
    """
    m: dict[str, str] = {}
    for doc in all_docs:
        cid  = doc["id"]
        name = doc["display_name"].lower()   # e.g. "amara and north enclave"
        # Add each word ≥4 chars as a keyword
        for word in re.split(r'\s+', name):
            if len(word) >= 4:
                m[word] = cid
        # Add the full name too
        m[name] = cid
    return m


def _detect_property_cols(query: str, all_ids: List[str], all_docs: List[dict]) -> List[str]:
    """
    If the query mentions a specific property name, return its collection ID(s).
    Returns empty list if no property name detected.
    """
    global _PROP_NAME_MAP
    if not _PROP_NAME_MAP:
        _PROP_NAME_MAP = _build_prop_name_map(all_docs)

    lower  = query.lower()
    matched: dict[str, bool] = {}
    for kw, cid in _PROP_NAME_MAP.items():
        if kw in lower and cid in all_ids:
            matched[cid] = True
    return list(matched.keys())


def _resolve_target_ids(
    query: str,
    raw_target_ids: List[str],
    all_docs: List[dict],
) -> tuple[List[str], str]:
    """
    Smart routing: given the full candidate list and the user query,
    return (final_target_ids, routing_reason).

    Priority order:
      1. If only 1 or 2 IDs given by caller → use as-is (user already filtered)
      2. If query is generic/company-level → route to generic info doc only
      3. If query mentions a specific property name → route to that property only
      4. If comparison query → use all IDs (keep existing logic)
      5. Otherwise → use all IDs (existing multi-col semantic search)
    """
    # Caller already narrowed it down
    if len(raw_target_ids) <= 2:
        return raw_target_ids, "caller-specified"

    # Generic company query → only generic doc
    if _GENERIC_COMPANY_PAT.search(query):
        generic = _find_generic_col(raw_target_ids)
        if generic:
            log.info("ROUTING: generic company query → %s", generic)
            return [generic], "generic-company"

    # Query mentions a specific property by name → that property only
    prop_cols = _detect_property_cols(query, raw_target_ids, all_docs)
    if prop_cols and not _is_comparison_query(query):
        log.info("ROUTING: property name detected → %s", prop_cols)
        return prop_cols, "property-name-match"

    # Comparison or open → all collections
    return raw_target_ids, "all-collections"


def _is_broad_single_query(query: str) -> bool:
    lower = query.strip().lower().rstrip("?!. ")
    bare_overview = {
        "details", "detail", "info", "information", "overview", "summary",
        "project details", "project info", "full details", "all details",
        "about", "describe", "project", "everything",
        "વિગત", "માહિતી", "વિગતો", "સારાંશ",
        "विवरण", "जानकारी", "सारांश", "पूरी जानकारी",
    }
    if lower in bare_overview:
        return True
    broad_patterns = [
        r"\b(project\s+and\s+price|project\s+detail|project\s+info|full\s+detail|"
        r"all\s+detail|complete\s+detail|full\s+info|overview|specifications?|"
        r"\bspec\b|project\s+features?|all\s+about)\b",
        r"(tell\s+me\s+about|describe\s+the|what\s+is\s+the\s+project|"
        r"project\s+ke\s+bare|project\s+ni\s+mahiti)",
        r"\bproject\b.{0,20}\bdetail",
    ]
    return any(re.search(p, lower) for p in broad_patterns)


_FOCUSED_N = 3


def rag_retrieve(collection_id: str, query: str, n: int = None) -> List[str]:
    try:
        col = _chroma.get_collection(name=collection_id)
    except Exception:
        return []
    count = col.count()
    if count == 0:
        return []

    if _is_broad_single_query(query):
        res    = col.get(include=["documents"])
        chunks = res.get("documents", [])
        log.info("RAG single-col %s: BROAD — ALL %d chunks (~%d chars)",
                 collection_id, len(chunks), sum(len(c) for c in chunks))
        return chunks

    k      = n or _FOCUSED_N
    vector = _get_query_embedding(query)
    res    = col.query(query_embeddings=[vector], n_results=min(k, count))
    chunks = res.get("documents", [[]])[0]
    log.info("RAG single-col %s: FOCUSED — semantic top-%d chunks (~%d chars)",
             collection_id, len(chunks), sum(len(c) for c in chunks))
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
#  SECTION 6 — RERA NORMALISER (for UI display only)
# ─────────────────────────────────────────────────────────────────────────────

_BROKEN_RERA_PAT = re.compile(
    r'(?<![A-Za-z0-9])([A-Z]\s+[A-Z])\s*,\s*([A-Z]\s+[A-Z])\s*,'
    r'\s*([A-Za-z0-9][A-Za-z0-9 ]*?(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9 ]*?){2,8})'
    r'(?=[।\.\!\?\n"\u0A00-\u0AFF\u0900-\u097F]|$)',
    re.IGNORECASE)

def _fix_broken_rera(m: re.Match) -> str:
    parts  = re.split(r'\s*,\s*', m.group(0))
    tokens = [re.sub(r'\s+', '', p.strip()).upper() for p in parts if p.strip()]
    return '/'.join(t for t in tokens if t)

def normalise_rera(text: str) -> str:
    return _BROKEN_RERA_PAT.sub(_fix_broken_rera, text)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — LLM
#
#  SIMPLIFIED: LLM now outputs a SINGLE plain-text response (no dual fields,
#  no TTS number conversion). Numbers stay as raw digits/text in the output.
#  The num2words_tts() engine in Section 2 handles all number conversion
#  deterministically before TTS.
# ─────────────────────────────────────────────────────────────────────────────

_openai = AsyncOpenAI(api_key=cfg.openai_api_key)

_SYSTEM_PROMPT_TEMPLATE = """You are Priya, a warm property sales advisor for Pacifica Companies.

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
When asked for lowest/highest/cheapest price across properties:
- Scan the PROPERTY CONTEXT for all prices per BHK type.
- State the winner property and its exact price as written in the context.
- Example: "The lowest 2 BHK price is at Madrid County starting from Rs. 41.14 lakhs."

ONBOARDING:
- __GREETING__: If name unknown → introduce as Priya, ask name. If name known → welcome back by name.
- When name given → thank them, ask phone number.
- Once name+phone collected → never ask again.

RERA FORMAT:
Write RERA exactly as in source: PR/GJ/GANDHINAGAR/AUDA/RAA04324/A1R/211021
Never break it or space it in your response.

NUMBER FORMAT:
- Write prices exactly as in source: "Rs. 44.5 lakhs", "₹ 75 lakhs", "2.5 crores"
- Write sq ft as: "1200 sq ft" or "1200 sqft"
- Write phone numbers as digits: "9876543210"
- Do NOT convert any numbers to words — write raw numbers only.

STYLE:
- Warm, natural sentences. No bullet points.
- 3–5 sentences max unless user asks for more.
- End with a gentle follow-up question when appropriate.

════════════════════════════════════════
OUTPUT FORMAT — PLAIN TEXT
════════════════════════════════════════
Respond with ONLY plain text in {language_upper}.
No JSON. No markdown. No bullet points. No extra formatting.
Your response will be shown directly to the user and also converted to speech.

════════════════════════════════════════
LANGUAGE
════════════════════════════════════════
Write your entire response in: {language_upper}
Exception: RERA codes stay in English slash format (PR/GJ/...).
Keep prices and numbers as raw digits (e.g. 75 lakhs, not seventy-five lakhs)."""

_LANG_INSTRUCTION = {
    "english":  "MANDATORY: Respond in English only.",
    "hindi":    "अनिवार्य: पूरा जवाब हिंदी में दें।",
    "gujarati": "ફરજિયાત: સંપૂર્ણ જવાબ ગુજરાતીમાં આપો.",
    "telugu":   "తప్పనిసరి: పూర్తి సమాధానం తెలుగులో ఇవ్వండి.",
    "tamil":    "கட்டாயம்: முழு பதிலையும் தமிழில் மட்டுமே தரவும்.",
}

_LANG_STARTER = {
    "gujarati": "જી, ",
    "hindi":    "जी, ",
    "telugu":   "అవును, ",
    "tamil":    "ஆம், ",
    "english":  "",
}


def _build_messages(
    query: str,
    output_language: str,
    chat_history: Optional[List[dict]],
    context_chunks: List[str] = None,
    context_map: Optional[dict] = None,
    user_name: str = "",
    user_phone: str = "",
) -> list:

    is_onboard = query.strip() in {"__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}
    lang       = output_language.lower()

    # ── Build context string ─────────────────────────────────────────────────
    if is_onboard:
        context_str = "[ONBOARDING — NO PROPERTY CONTEXT NEEDED]"
    elif context_map:
        parts = []
        for col_id, chunks in context_map.items():
            if chunks:
                prop_name = col_id.replace("_col", "").replace("_", " ").title()
                parts.append(f"=== {prop_name} ===\n" + "\n\n".join(chunks))
        context_str = "\n\n".join(parts) if parts else "[NO_CONTEXT_AVAILABLE]"
    elif context_chunks:
        context_str = "\n\n---\n\n".join(context_chunks)
    else:
        context_str = "[NO_CONTEXT_AVAILABLE]"

    # ── Assemble system prompt ────────────────────────────────────────────────
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        language_upper=lang.upper(),
    )

    user_ctx  = f"\nUser name: {user_name}" if user_name else "\nUser name: NOT YET COLLECTED"
    user_ctx += (f"\nUser phone: {user_phone} (collected — never ask again)"
                 if user_phone else "\nUser phone: NOT YET COLLECTED")

    system_content = (
        f"{system_prompt}"
        f"\n\n{user_ctx}"
        f"\n\nPROPERTY CONTEXT:\n{context_str}"
    )

    messages = [{"role": "system", "content": system_content}]

    # ── Inject chat history ───────────────────────────────────────────────────
    if chat_history:
        for h in chat_history[-8:]:
            messages.append({"role": h["role"], "content": h["content"]})

    # ── User message ──────────────────────────────────────────────────────────
    lang_note = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["english"])
    messages.append({
        "role": "user",
        "content": (
            f"[{lang_note}]\n"
            f"[NUMBERS: Write all numbers as raw digits/notation — e.g. 75 lakhs, not seventy-five]\n"
            f"{query}"
        )
    })

    starter = _LANG_STARTER.get(lang, "")
    if starter:
        messages.append({"role": "assistant", "content": starter})

    return messages


async def llm_answer(
    query: str,
    output_language: str = "english",
    chat_history: Optional[List[dict]] = None,
    context_chunks: List[str] = None,
    context_map: Optional[dict] = None,
    user_name: str = "",
    user_phone: str = "",
) -> tuple[str, str]:
    """
    Single LLM call. Returns (ui_text, tts_text).
    ui_text  = plain LLM response (raw numbers, shown to user)
    tts_text = num2words_tts(ui_text) — deterministic number→words conversion
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
    )
    raw = resp.choices[0].message.content.strip()
    log.info("LLM OUTPUT (raw): %s", raw[:500])
    log.info("LLM OUTPUT — tokens: prompt=%d completion=%d total=%d",
             resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens)
    log.info("=" * 70)

    # Fix any broken RERA codes in UI text
    ui_text = normalise_rera(raw)

    # Deterministic number conversion for TTS
    lang     = output_language.lower()
    tts_text = num2words_tts(ui_text, lang)

    log.info("UI  TEXT: %s", ui_text[:300])
    log.info("TTS TEXT: %s", tts_text[:300])

    return ui_text, tts_text

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — TTS
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
    """Strip markdown symbols and fix punctuation for TTS."""
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
#  SECTION 9 — FASTAPI
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
                 r.get("filename", "?"), r.get("status", "?"), r.get("chunks", 0))
    yield
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    log.info("Shutdown complete.")

app = FastAPI(title="PropVoice — Pacifica Companies RAG API", version="5.0.0",
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
    session_id:      str              = "default"
    pdf_id:          Optional[str]    = None
    pdf_ids:         Optional[List[str]] = None
    output_language: str              = "english"

class ChatResponse(BaseModel):
    answer_text:     str
    audio_base64:    Optional[str]    = None
    pdf_id:          Optional[str]    = None
    pdf_ids:         Optional[List[str]] = None
    output_language: str


async def _handle_query(
    query: str,
    session_id: str,
    output_language: str,
    pdf_id: str = "",
    pdf_ids_str: str = "",
    pdf_ids_list: Optional[List[str]] = None,
) -> ChatResponse:
    user    = db_get_user(session_id)
    u_name  = user.get("name", "")
    u_phone = user.get("phone", "")

    ONBOARD = {"__GREETING__", "__ASK_PHONE__", "__PHONE_SKIP__", "__PHONE_DONE__"}

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

    # Replace the existing if/elif block:
    if query in ONBOARD:
        pass
    elif len(target_ids) == 1:
        context_chunks = await loop.run_in_executor(
            None, rag_retrieve, target_ids[0], query)
    elif len(target_ids) > 1:
        all_docs = rag_list_docs()
        routed_ids, reason = _resolve_target_ids(query, target_ids, all_docs)
        log.info("ROUTING: '%s...' → %d col(s) [%s]", query[:50], len(routed_ids), reason)
        if len(routed_ids) == 1:
            context_chunks = await loop.run_in_executor(
                None, rag_retrieve, routed_ids[0], query)
        else:
            context_map = await loop.run_in_executor(
                None, lambda: rag_retrieve_multi(routed_ids, query))

    history = db_get_history(session_id, limit=10)
    db_save_turn(session_id, "user", query, output_language, ",".join(target_ids))

    ui_text, tts_text = await llm_answer(
        query, output_language, history,
        context_chunks, context_map, u_name, u_phone)

    # Store plain ui_text in history (no JSON wrapping needed anymore)
    db_save_turn(session_id, "assistant", ui_text, output_language, ",".join(target_ids))

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
    return {"message": "PropVoice API v5.0 running.", "docs": "/docs"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "5.0.0"}

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
    return {"history": db_get_history(session_id, limit=50)}

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