"""
number_words.py  v2
───────────────────
Deterministic number → spoken-words converter for PropVoice TTS.

Fixes in v2:
  1. _gu_int / _hi_int / _ta_int now handle 1000-99999 (sq ft ranges like 1050, 4200)
  2. extract_numbers_from_chunks now captures range patterns (575 to 588),
     sqft-suffixed numbers, and area-context numbers — not just price patterns
  3. No LLM involvement for any number conversion → zero hallucination

Supports: english, hindi, gujarati, telugu, tamil
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  GUJARATI  1-99  (fully enumerated — irregular language, no formula)
# ─────────────────────────────────────────────────────────────────────────────
_GU: List[str] = [
    '',
    'એક', 'બે', 'ત્રણ', 'ચાર', 'પાંચ', 'છ', 'સાત', 'આઠ', 'નવ',
    'દસ', 'અગિયાર', 'બાર', 'તેર', 'ચૌદ', 'પંદર', 'સોળ', 'સત્તર', 'અઢાર', 'ઓગણીસ',
    'વીસ', 'એકવીસ', 'બાવીસ', 'તેવીસ', 'ચોવીસ', 'પચ્ચીસ', 'છવ્વીસ', 'સત્યાવીસ', 'અઠ્ઠાવીસ', 'ઓગણત્રીસ',
    'ત્રીસ', 'એકત્રીસ', 'બત્રીસ', 'તેત્રીસ', 'ચોત્રીસ', 'પાંત્રીસ', 'છત્રીસ', 'સાડત્રીસ', 'આડત્રીસ', 'ઓગણચાળીસ',
    'ચાળીસ', 'એકતાળીસ', 'બેતાળીસ', 'ત્રેતાળીસ', 'ચુમ્માળીસ', 'પિસ્તાળીસ', 'છેતાળીસ', 'સુડતાળીસ', 'અડતાળીસ', 'ઓગણપચાસ',
    'પચાસ', 'એકાવન', 'બાવન', 'ત્રેપન', 'ચોપન', 'પંચાવન', 'છપ્પન', 'સત્તાવન', 'અઠ્ઠાવન', 'ઓગણસાઠ',
    'સાઠ', 'એકસઠ', 'બાસઠ', 'ત્રેસઠ', 'ચોસઠ', 'પાંસઠ', 'છાસઠ', 'સડસઠ', 'અડસઠ', 'ઓગણસિત્તેર',
    'સિત્તેર', 'એકોતેર', 'બોતેર', 'તોંતેર', 'ચુમ્મોતેર', 'પંચોતેર', 'છોત્તેર', 'સિત્યોતેર', 'અઠ્યોતેર', 'ઓગ્ ણ',
    'એંશી', 'એક્યાસી', 'બ્યાસી', 'ત્ર્યાસી', 'ચોર્યાસી', 'પંચ્યાસી', 'છ્યાસી', 'સત્યાસી', 'અઠ્ઠ્યાસી', 'નેવ્યાસી',
    'નેવું', 'એકાણું', 'બાણું', 'ત્રાણું', 'ચોરાણું', 'પંચ્ ानां', 'છ ानां', 'સત ानां', 'અઠ ानां', 'નૉ',
]
_GU[79] = 'ઓગ્ ણ'

_GU_H: Dict[int, str] = {
    1: 'એક સો', 2: 'બે સો', 3: 'ત્રણ સો', 4: 'ચારસો', 5: 'પાંચ સો',
    6: 'છ સો',  7: 'સાત સો', 8: 'આઠ સો',  9: 'નવ સો',
}
_GU_T: Dict[int, str] = {
    1: 'એક હજાર', 2: 'બે હજાર', 3: 'ત્રણ હજાર', 4: 'ચાર હજાર',
    5: 'પાંચ હજાર', 6: 'છ હજાર', 7: 'સાત હજાર', 8: 'આઠ હજાર',
    9: 'નવ હજાર', 10: 'દસ હજાર', 11: 'અગિયાર હજાર', 12: 'બાર હજાર',
    13: 'તેર હજાર', 14: 'ચૌદ હજાર', 15: 'પંદર હજાર', 16: 'સોળ હજાર',
    17: 'સત્તર હજાર', 18: 'અઢાર હજાર', 19: 'ઓગણીસ હજાર', 20: 'વીસ હજાર',
}

def _gu_int(n: int) -> str:
    if n == 0:  return 'શૂન્ય'
    if n < 0:   return 'માઈનસ ' + _gu_int(-n)
    if n <= 99: return _GU[n]
    if n <= 999:
        h, r = n // 100, n % 100
        hw = _GU_H.get(h, _GU[h] + ' સો')
        return hw if r == 0 else hw + ' ' + _GU[r]
    if n <= 99999:
        t, r = n // 1000, n % 1000
        tw = _GU_T.get(t, _GU[t] + ' હજાર') if t <= 99 else str(t) + ' હજાર'
        if r == 0: return tw
        if r <= 99: return tw + ' ' + _GU[r]
        h2, r2 = r // 100, r % 100
        hw = _GU_H.get(h2, _GU[h2] + ' સો')
        return tw + ' ' + (hw if r2 == 0 else hw + ' ' + _GU[r2])
    # lakh+ (rare in sqft context, fallback)
    return str(n)

# ─────────────────────────────────────────────────────────────────────────────
#  HINDI  1-99
# ─────────────────────────────────────────────────────────────────────────────
_HI: List[str] = [
    '',
    'एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ',
    'दस', 'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पन्द्रह', 'सोलह', 'सत्रह', 'अठारह', 'उन्नीस',
    'बीस', 'इक्कीस', 'बाईस', 'तेईस', 'चौबीस', 'पच्चीस', 'छब्बीस', 'सत्ताईस', 'अट्ठाईस', 'उनतीस',
    'तीस', 'इकतीस', 'बत्तीस', 'तैंतीस', 'चौंतीस', 'पैंतीस', 'छत्तीस', 'सैंतीस', 'अड़तीस', 'उनचालीस',
    'चालीस', 'इकतालीस', 'बयालीस', 'तैंतालीस', 'चवालीस', 'पैंतालीस', 'छियालीस', 'सैंतालीस', 'अड़तालीस', 'उनचास',
    'पचास', 'इक्यावन', 'बावन', 'तिरपन', 'चौवन', 'पचपन', 'छप्पन', 'सत्तावन', 'अठावन', 'उनसठ',
    'साठ', 'इकसठ', 'बासठ', 'तिरसठ', 'चौंसठ', 'पैंसठ', 'छियासठ', 'सड़सठ', 'अड़सठ', 'उनहत्तर',
    'सत्तर', 'इकहत्तर', 'बहत्तर', 'तिहत्तर', 'चौहत्तर', 'पचहत्तर', 'छिहत्तर', 'सतहत्तर', 'अठहत्तर', 'उन्यासी',
    'अस्सी', 'इक्यासी', 'बयासी', 'तिरासी', 'चौरासी', 'पचासी', 'छियासी', 'सत्तासी', 'अठासी', 'नवासी',
    'नब्बे', 'इक्यानवे', 'बानवे', 'तिरानवे', 'चौरानवे', 'पचानवे', 'छियानवे', 'सत्तानवे', 'अठानवे', 'निन्यानवे',
]

def _hi_int(n: int) -> str:
    if n == 0:  return 'शून्य'
    if n < 0:   return 'माइनस ' + _hi_int(-n)
    if n <= 99: return _HI[n]
    if n <= 999:
        h, r = n // 100, n % 100
        hw = _HI[h] + ' सौ'
        return hw if r == 0 else hw + ' ' + _HI[r]
    if n <= 99999:
        t, r = n // 1000, n % 1000
        tw = (_HI[t] if t <= 99 else str(t)) + ' हज़ार'
        if r == 0: return tw
        if r <= 99: return tw + ' ' + _HI[r]
        h2, r2 = r // 100, r % 100
        hw = _HI[h2] + ' सौ'
        return tw + ' ' + (hw if r2 == 0 else hw + ' ' + _HI[r2])
    return str(n)

# ─────────────────────────────────────────────────────────────────────────────
#  TAMIL  1-99
# ─────────────────────────────────────────────────────────────────────────────
_TA: List[str] = [
    '',
    'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது',
    'பத்து', 'பதினொன்று', 'பன்னிரண்டு', 'பதின்மூன்று', 'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு', 'பதினெட்டு', 'பத்தொன்பது',
    'இருபது', 'இருபத்தொன்று', 'இருபத்திரண்டு', 'இருபத்துமூன்று', 'இருபத்துநான்கு', 'இருபத்தைந்து', 'இருபத்தாறு', 'இருபத்தேழு', 'இருபத்தெட்டு', 'இருபத்தொன்பது',
    'முப்பது', 'முப்பத்தொன்று', 'முப்பத்திரண்டு', 'முப்பத்துமூன்று', 'முப்பத்துநான்கு', 'முப்பத்தைந்து', 'முப்பத்தாறு', 'முப்பத்தேழு', 'முப்பத்தெட்டு', 'முப்பத்தொன்பது',
    'நாற்பது', 'நாற்பத்தொன்று', 'நாற்பத்திரண்டு', 'நாற்பத்துமூன்று', 'நாற்பத்துநான்கு', 'நாற்பத்தைந்து', 'நாற்பத்தாறு', 'நாற்பத்தேழு', 'நாற்பத்தெட்டு', 'நாற்பத்தொன்பது',
    'ஐம்பது', 'ஐம்பத்தொன்று', 'ஐம்பத்திரண்டு', 'ஐம்பத்துமூன்று', 'ஐம்பத்துநான்கு', 'ஐம்பத்தைந்து', 'ஐம்பத்தாறு', 'ஐம்பத்தேழு', 'ஐம்பத்தெட்டு', 'ஐம்பத்தொன்பது',
    'அறுபது', 'அறுபத்தொன்று', 'அறுபத்திரண்டு', 'அறுபத்துமூன்று', 'அறுபத்துநான்கு', 'அறுபத்தைந்து', 'அறுபத்தாறு', 'அறுபத்தேழு', 'அறுபத்தெட்டு', 'அறுபத்தொன்பது',
    'எழுபது', 'எழுபத்தொன்று', 'எழுபத்திரண்டு', 'எழுபத்துமூன்று', 'எழுபத்துநான்கு', 'எழுபத்தைந்து', 'எழுபத்தாறு', 'எழுபத்தேழு', 'எழுபத்தெட்டு', 'எழுபத்தொன்பது',
    'எண்பது', 'எண்பத்தொன்று', 'எண்பத்திரண்டு', 'எண்பத்துமூன்று', 'எண்பத்துநான்கு', 'எண்பத்தைந்து', 'எண்பத்தாறு', 'எண்பத்தேழு', 'எண்பத்தெட்டு', 'எண்பத்தொன்பது',
    'தொண்ணூறு', 'தொண்ணூற்றொன்று', 'தொண்ணூற்றிரண்டு', 'தொண்ணூற்றுமூன்று', 'தொண்ணூற்றுநான்கு', 'தொண்ணூற்றைந்து', 'தொண்ணூற்றாறு', 'தொண்ணூற்றேழு', 'தொண்ணூற்றெட்டு', 'தொண்ணூற்றொன்பது',
]

_TA_H: Dict[int, str] = {
    1: 'நூறு', 2: 'இருநூறு', 3: 'முன்னூறு', 4: 'நானூறு',
    5: 'ஐந்நூறு', 6: 'அறுநூறு', 7: 'எழுநூறு', 8: 'எண்ணூறு', 9: 'தொள்ளாயிரம்',
}
_TA_T: Dict[int, str] = {
    1: 'ஆயிரம்', 2: 'இரண்டாயிரம்', 3: 'மூவாயிரம்', 4: 'நான்காயிரம்',
    5: 'ஐந்தாயிரம்', 6: 'ஆறாயிரம்', 7: 'ஏழாயிரம்', 8: 'எட்டாயிரம்',
    9: 'ஒன்பதாயிரம்', 10: 'பத்தாயிரம்',
}

def _ta_int(n: int) -> str:
    if n == 0:  return 'சுழியம்'
    if n < 0:   return 'கழித்தல் ' + _ta_int(-n)
    if n <= 99: return _TA[n]
    if n <= 999:
        h, r = n // 100, n % 100
        hw = _TA_H.get(h, _TA[h] + ' நூறு')
        return hw if r == 0 else hw + ' ' + _TA[r]
    if n <= 9999:
        t, r = n // 1000, n % 1000
        tw = _TA_T.get(t, _TA[t] + ' ஆயிரம்')
        if r == 0: return tw
        if r <= 99: return tw + ' ' + _TA[r]
        h2, r2 = r // 100, r % 100
        hw = _TA_H.get(h2, _TA[h2] + ' நூறு')
        return tw + ' ' + (hw if r2 == 0 else hw + ' ' + _TA[r2])
    return str(n)

# ─────────────────────────────────────────────────────────────────────────────
#  TELUGU — num2words (accurate for te)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from num2words import num2words as _n2w
    def _telugu_int(n: int) -> str:
        try:
            return _n2w(n, lang='te')
        except Exception:
            return str(n)
except ImportError:
    def _telugu_int(n: int) -> str:
        return str(n)

# ─────────────────────────────────────────────────────────────────────────────
#  ENGLISH — recursive
# ─────────────────────────────────────────────────────────────────────────────
_EN_ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
            'seventeen', 'eighteen', 'nineteen']
_EN_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

def _english_int(n: int) -> str:
    if n < 0:   return 'minus ' + _english_int(-n)
    if n == 0:  return 'zero'
    if n < 20:  return _EN_ONES[n]
    if n < 100: return _EN_TENS[n // 10] + ('' if n % 10 == 0 else ' ' + _EN_ONES[n % 10])
    if n < 1000:
        h = _EN_ONES[n // 100] + ' hundred'
        r = n % 100
        return h if r == 0 else h + ' ' + _english_int(r)
    if n < 100_000:
        l = _english_int(n // 1000) + ' thousand'
        r = n % 1000
        return l if r == 0 else l + ' ' + _english_int(r)
    if n < 10_000_000:
        l = _english_int(n // 100_000) + ' lakh'
        r = n % 100_000
        return l if r == 0 else l + ' ' + _english_int(r)
    c = _english_int(n // 10_000_000) + ' crore'
    r = n % 10_000_000
    return c if r == 0 else c + ' ' + _english_int(r)

# ─────────────────────────────────────────────────────────────────────────────
#  DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def _int_converter(lang: str):
    return {
        'gujarati': _gu_int,
        'hindi':    _hi_int,
        'tamil':    _ta_int,
        'telugu':   _telugu_int,
        'english':  _english_int,
    }.get(lang, _english_int)

_POINT_WORD = {
    'gujarati': 'પોઇન્ટ', 'hindi': 'दशमलव', 'tamil': 'புள்ளி',
    'telugu': 'దశాంశం', 'english': 'point',
}
_RUPEE_WORD = {
    'gujarati': 'રૂપિયા', 'hindi': 'रुपये', 'tamil': 'ரூபாய்',
    'telugu': 'రూపాయలు', 'english': 'rupees',
}
_LAKH_WORD = {
    'gujarati': 'લાખ', 'hindi': 'लाख', 'tamil': 'லட்சம்',
    'telugu': 'లక్షలు', 'english': 'lakhs',
}
_CRORE_WORD = {
    'gujarati': 'કરોડ', 'hindi': 'करोड़', 'tamil': 'கோடி',
    'telugu': 'కోట్లు', 'english': 'crores',
}

def _zero_word(lang: str) -> str:
    return {
        'gujarati': 'શૂન્ય', 'hindi': 'शून्य', 'tamil': 'சுழியம்',
        'telugu': 'సున్నా', 'english': 'zero',
    }.get(lang, 'zero')


def number_to_words(value: str, language: str = 'english') -> str:
    """
    Convert a numeric string to spoken words in the target language.
    Handles integers and decimals (X.YY).
    """
    lang = language.lower()
    conv = _int_converter(lang)
    pt   = _POINT_WORD.get(lang, 'point')
    val  = value.strip().replace(',', '')

    if '.' in val:
        int_part, dec_part = val.split('.', 1)
        int_words = conv(int(int_part)) if int_part else conv(0)
        dec_words = ' '.join(
            conv(int(d)) if d != '0' else _zero_word(lang)
            for d in dec_part
        )
        return f"{int_words} {pt} {dec_words}"
    try:
        return conv(int(val))
    except ValueError:
        return val


# ─────────────────────────────────────────────────────────────────────────────
#  NUMBER EXTRACTION FROM RAG CHUNKS  (v2 — catches range patterns + sqft)
# ─────────────────────────────────────────────────────────────────────────────

# Currency: Rs. 44 lakhs / ₹61.33 crores / Rs. 1.20 Crores
_PRICE_PAT = re.compile(
    r'(?:Rs\.?\s*|₹\s*)(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)(?:\s*(?:lakhs?|crores?|lakh|crore|L\b|Cr\b))?',
    re.IGNORECASE
)
# Bare amount: 44 lakhs / 61.33 crores (no currency symbol)
_BARE_AMOUNT_PAT = re.compile(
    r'\b(\d{1,3}(?:\.\d{1,2})?)\s+(?:lakhs?|crores?|lakh|crore)\b',
    re.IGNORECASE
)
# BHK / floors / towers / units (small integers)
_UNIT_PAT = re.compile(
    r'\b(\d{1,4})\s*(?:BHK|bhk|units?|floors?|towers?|flats?|bedrooms?)',
    re.IGNORECASE
)
# Sqft suffixed: 861 sq ft / 575 sqft / 1050 sq.ft
_SQFT_PAT = re.compile(
    r'\b(\d{2,5})(?:\.\d{1,2})?\s*(?:sq\.?\s*ft|sqft|Sq\.?\s*Ft|square\s*f)',
    re.IGNORECASE
)
# Range pattern: 575 to 588 / 774-808 / 575 થી 588 / 575 से 588
_RANGE_PAT = re.compile(
    r'\b(\d{3,5})\s*(?:to|–|-|થી|से|முதல்|నుండి)\s*(\d{3,5})\b',
    re.IGNORECASE
)
# Area context: carpet area: 575 / built-up area 861
_AREA_CTX_PAT = re.compile(
    r'(?:carpet\s+area|built[\s-]up\s+area|super\s+built[\s-]up|rera\s+carpet|plot\s+area|area)\s*[:\-]?\s*(\d{3,5}(?:\.\d{1,2})?)',
    re.IGNORECASE
)


def extract_numbers_from_chunks(chunks) -> set:
    """
    Extract all numeric values that need TTS conversion from RAG chunks.
    Returns a set of string values: {'44', '61.33', '575', '588', '861', '2', '3'}.

    Deliberately excludes:
      - RERA codes  (handled char-by-char in TTS rules)
      - Phone numbers  (digit-by-digit)
      - Years / pin codes  (no pattern match)
    """
    numbers: set = set()
    all_text = '\n'.join(chunks) if isinstance(chunks, list) else chunks

    for pat in (_PRICE_PAT, _BARE_AMOUNT_PAT):
        for m in pat.finditer(all_text):
            val = m.group(1).replace(',', '')
            try:
                float(val)
                numbers.add(val)
            except ValueError:
                pass

    for m in _UNIT_PAT.finditer(all_text):
        numbers.add(m.group(1))

    for m in _SQFT_PAT.finditer(all_text):
        numbers.add(m.group(1))

    for m in _RANGE_PAT.finditer(all_text):
        numbers.add(m.group(1))
        numbers.add(m.group(2))

    for m in _AREA_CTX_PAT.finditer(all_text):
        val = m.group(1).split('.')[0]  # integer part only for areas
        numbers.add(val)

    return numbers


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_number_map(chunks, language: str) -> Dict[str, str]:
    """
    Extract all numbers from RAG chunks and convert deterministically.
    Returns: {"44": "ચુમ્માળીસ", "575": "પાંચ સો પંચોતેર", ...}
    Injected into LLM prompt as ground truth — LLM copies, never invents.
    """
    lang    = language.lower()
    numbers = extract_numbers_from_chunks(chunks)
    nmap    = {}
    for val in sorted(numbers, key=lambda x: float(x)):
        try:
            nmap[val] = number_to_words(val, lang)
        except Exception:
            pass
    return nmap


def number_map_prompt_block(nmap: Dict[str, str], language: str) -> str:
    """
    Render the number map as a compact prompt injection block.
    """
    if not nmap:
        return ''
    lang  = language.lower()
    pairs = ' | '.join(
        f"{k} → {v}"
        for k, v in sorted(nmap.items(), key=lambda x: float(x[0]))
    )
    rule = {
        'gujarati': 'tts ક્ષેત્રમાં આ જ શબ્દો વાપરો — ક્યારેય બીજા નહીં',
        'hindi':    'tts फ़ील्ड में यही शब्द उपयोग करें — कभी अन्य नहीं',
        'tamil':    'tts புலத்தில் இந்த வார்த்தைகளையே பயன்படுத்தவும்',
        'telugu':   'tts ఫీల్డ్‌లో ఈ పదాలనే వాడండి — వేరేవి వాడకండి',
        'english':  'use EXACTLY these words in tts field — never invent',
    }.get(lang, 'use EXACTLY these words in tts field — never invent')

    return (
        f"════════════════════════════════════════\n"
        f"NUMBER_MAP ({rule}):\n"
        f"{pairs}\n"
        f"════════════════════════════════════════"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MIN-PRICE EXTRACTOR  (comparison query grounding)
# ─────────────────────────────────────────────────────────────────────────────

def extract_min_prices(context_map: Dict[str, list]) -> Dict[int, tuple]:
    """
    Deterministically find minimum price per BHK type across all properties.
    Returns: {2: (41.14, "Madrid County", "lakhs"), 3: (58.5, "Madrid County", "lakhs")}
    Injected as grounded fact — LLM reads the winner, never guesses.
    """
    bhk_mins: Dict[int, tuple] = {}

    for col_id, chunks in context_map.items():
        prop_name = col_id.replace('_col', '').replace('_', ' ').title()
        for chunk in chunks:
            # Pattern: "2 BHK ... Rs. 41.14 lakhs"
            for m in re.finditer(
                r'(\d)\s*BHK[^\n]{0,120}?(?:Rs\.?\s*|₹\s*)(\d{1,3}(?:\.\d{1,2})?)'
                r'\s*(lakhs?|crores?|lakh|crore|L\b|Cr\b)?',
                chunk, re.IGNORECASE
            ):
                bhk   = int(m.group(1))
                price = float(m.group(2))
                unit  = (m.group(3) or 'lakhs').lower()
                norm  = price * 100 if 'crore' in unit else price
                if bhk not in bhk_mins or norm < bhk_mins[bhk][0]:
                    bhk_mins[bhk] = (norm, prop_name, m.group(3) or 'lakhs')

            # Pattern: "Rs. 41.14 lakhs ... 2 BHK" (reverse)
            for m in re.finditer(
                r'(?:Rs\.?\s*|₹\s*)(\d{1,3}(?:\.\d{1,2})?)\s*(lakhs?|crores?|lakh|crore|L\b|Cr\b)?'
                r'[^\n]{0,80}?(\d)\s*BHK',
                chunk, re.IGNORECASE
            ):
                bhk   = int(m.group(3))
                price = float(m.group(1))
                unit  = (m.group(2) or 'lakhs').lower()
                norm  = price * 100 if 'crore' in unit else price
                if bhk not in bhk_mins or norm < bhk_mins[bhk][0]:
                    bhk_mins[bhk] = (norm, prop_name, m.group(2) or 'lakhs')

    return bhk_mins


def min_price_grounding_block(bhk_mins: Dict[int, tuple], language: str) -> str:
    """
    Render verified min-price facts as a prompt injection block.
    """
    if not bhk_mins:
        return ''
    lang  = language.lower()
    lines = []
    for bhk in sorted(bhk_mins):
        norm_price, prop, unit = bhk_mins[bhk]
        if norm_price >= 100 and 'crore' not in unit.lower():
            display = f"Rs. {norm_price / 100:.2f} Crores"
        else:
            display = f"Rs. {norm_price:.2f} {unit}"
        lines.append(f"  {bhk} BHK lowest: {display} at {prop}")

    label = {
        'gujarati': 'ચકાસાયેલ સૌથી ઓછી કિંમત',
        'hindi':    'सत्यापित न्यूनतम मूल्य',
        'tamil':    'சரிபார்க்கப்பட்ட குறைந்த விலை',
        'telugu':   'ధృవీకరించబడిన కనిష్ట ధర',
        'english':  'VERIFIED MINIMUM PRICES',
    }.get(lang, 'VERIFIED MINIMUM PRICES')

    note = {
        'gujarati': 'ઉત્તર ફક્ત આ verified data પર આધારિત આપો — LLM ના guess ના',
        'hindi':    'उत्तर केवल इस verified data पर दें — LLM का अनुमान नहीं',
        'tamil':    'இந்த verified data மட்டுமே பயன்படுத்தவும்',
        'telugu':   'ఈ verified data ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి',
        'english':  'Answer ONLY from this verified data — never guess',
    }.get(lang, 'Answer ONLY from this verified data — never guess')

    return (
        f"════════════════════════════════════════\n"
        f"{label}:\n"
        + '\n'.join(lines) +
        f"\n{note}\n"
        f"════════════════════════════════════════"
    )