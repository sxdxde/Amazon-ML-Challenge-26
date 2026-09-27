"""Country-agnostic, language-general text normalisation.

Everything here is a hand-written rule or dictionary (documented in docs/methodology_notes.md);
nothing is looked up externally. No country value is special-cased: the only use of the country
string is to drop a name token equal to the record's own country (e.g. "Alcon (India) Traders").

Pipeline per string:
  1. Brahmic-script transliteration to Latin (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
     Tamil, Telugu, Kannada, Malayalam share the ISCII-derived code-point layout, so one table
     indexed by offset-within-block covers all of them).
  2. NFKD + accent stripping + casefold (no unidecode: GPL).
  3. '&' / ' + ' -> ' and '; punctuation -> space; whitespace collapse.
  4. Token-level abbreviation canonicalisation (street terms, legal forms, common words).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# =============================================================================== transliteration
_BRAHMIC_LO, _BRAHMIC_HI = 0x0900, 0x0D7F
# Offset (within a 128-code-point Brahmic block) -> Latin. Consonants carry an inherent 'a'.
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_INDEP_VOWELS = {
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
    0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "a", 0x73: "i", 0x74: "u",
}
_MATRAS = {
    0x3A: "o", 0x3B: "a", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4E: "e", 0x4F: "aw", 0x55: "", 0x56: "", 0x57: "au", 0x62: "li", 0x63: "li",
}
_SIGNS = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h", 0x50: "om", 0x70: "n", 0x71: "",
          0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k", 0x64: " ", 0x65: " "}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NUKTA_MAP = {"j": "z", "f": "f", "k": "q", "d": "r", "dh": "rh", "g": "g", "kh": "kh"}
_RE_BRAHMIC = re.compile("[ऀ-ൿ]")
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def translit_brahmic(s: str) -> str:
    """Rough phonetic romanisation of any ISCII-layout Indic script (schwa deleted at word end)."""
    if not _RE_BRAHMIC.search(s):
        return s
    out: list[str] = []
    pending_a = False  # last emitted consonant still carries its inherent 'a'
    prev = -1  # previous Brahmic code point (for a few script-specific digraphs)

    def flush_a(at_word_end: bool) -> None:
        nonlocal pending_a
        if pending_a and not at_word_end:
            out.append("a")
        pending_a = False

    for ch in s:
        cp = ord(ch)
        if not (_BRAHMIC_LO <= cp <= _BRAHMIC_HI):
            flush_a(at_word_end=not ch.isalnum())
            out.append(ch)
            continue
        off = cp & 0x7F
        block = cp & ~0x7F
        if off in _CONS:
            if block == 0x0D00 and off == 0x31 and prev == cp - 0x31 + _VIRAMA and len(out) > 0 and out[-1] == "r":
                out[-1] = "t"  # Malayalam റ്റ = tt
                out.append("t")
                pending_a = True
                prev = cp
                continue
            if block == 0x0B80 and off == 0x2A and out and out[-1] == "h" and prev == 0x0B83:
                out[-1] = ""  # Tamil ஃப = f
                out.append("f")
                pending_a = True
                prev = cp
                continue
            flush_a(False)
            out.append(_CONS[off])
            pending_a = True
        elif off in _MATRAS:
            pending_a = False
            out.append(_MATRAS[off])
        elif off == _VIRAMA:
            pending_a = False
        elif off == _NUKTA:
            if out and out[-1] in _NUKTA_MAP:
                out[-1] = _NUKTA_MAP[out[-1]]
        elif off in _INDEP_VOWELS:
            flush_a(False)
            out.append(_INDEP_VOWELS[off])
        elif 0x66 <= off <= 0x6F:
            flush_a(True)
            out.append(str(off - 0x66))
        elif off in _SIGNS:
            flush_a(False)
            out.append(_SIGNS[off])
        # other code points (accents/stress marks) are dropped
        prev = cp
    flush_a(True)
    return "".join(out)


# =============================================================================== ascii folding
_SPECIAL = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o",
                          "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ı": "i", "þ": "th", "ð": "d",
                          "°": " ", "º": " ", "ª": " ", "’": "'", "‘": "'", "´": "'", "`": "'"})
_RE_AMP = re.compile(r"\s*&\s*|\s\+\s")
_RE_APOS = re.compile(r"(?<=\w)'(?=s\b)|'")  # "diana's" -> "dianas"
_RE_NONWORD = re.compile(r"[^0-9a-z]+")
_RE_NUMALPHA = re.compile(r"(?<=\d)(?=[a-z]{3,})|(?<=[a-z]{3})(?=\d)")  # "no10" stays, "10th" stays


def fold(s: str) -> str:
    """Transliterate, NFKD-strip accents, casefold, & -> and, punctuation -> space."""
    s = translit_brahmic(s.translate(_ZW))
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = _RE_AMP.sub(" and ", s)
    s = _RE_APOS.sub("", s)
    s = _RE_NONWORD.sub(" ", s)
    return " ".join(s.split())


def has_nonlatin(s: str) -> bool:
    return bool(_RE_BRAHMIC.search(s)) or any(ord(c) > 0x24F and c.isalpha() for c in s)


# =============================================================================== dictionaries
# Legal forms (EN / IN / FR / generic) -> canonical code. Transliterated Hindi forms included.
LEGAL = {
    "llc": "llc", "l l c": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
    "corporation": "corp", "co": "co", "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd",
    "limted": "ltd", "lmited": "ltd", "pvt": "pvt", "private": "pvt", "pvtltd": "pvt", "llp": "llp",
    "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc", "pa": "pa", "ltda": "ltd", "gmbh": "gmbh",
    "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa", "pty": "pty",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "scp": "scp", "selarl": "selarl", "sca": "sca", "gie": "gie",
    # romanised Devanagari/Indic spellings produced by translit_brahmic
    "praivet": "pvt", "praiveta": "pvt", "prayvet": "pvt", "limiteda": "ltd",
    "limitid": "ltd", "elelpi": "llp", "kanpani": "co", "kampani": "co", "karporeshan": "corp",
    "inkorporeted": "inc", "praibhet": "pvt", "piraivet": "pvt", "praivat": "pvt", "limitet": "ltd",
    "limittad": "ltd", "limitad": "ltd", "elaelapi": "llp", "elelapi": "llp", "praivatt": "pvt",
}
# Multi-token legal phrases collapsed before token lookup.
LEGAL_PHRASES = [(re.compile(p), r) for p, r in [
    (r"\bp\s+ltd\b", "pvt ltd"), (r"\bpvt\s*ltd\b", "pvt ltd"), (r"\bl\s+l\s+c\b", "llc"),
    (r"\bl\s+l\s+p\b", "llp"), (r"\bs\s+a\s+r\s+l\b", "sarl"), (r"\bs\s+a\s+s\s+u\b", "sasu"),
    (r"\bs\s+a\s+s\b", "sas"), (r"\be\s+u\s+r\s+l\b", "eurl"), (r"\bs\s+c\s+i\b", "sci"),
    (r"\bp\s+c\b", "pc"), (r"\bl\s+p\b", "lp"), (r"\bp\s+a\b", "pa"), (r"\bpra\s+li\b", "pvt ltd"),
    (r"\bp\s+l\s+c\b", "plc"), (r"\bs\s+a\b", "sa"),
]]

# Street / address vocabulary (EN, IN, FR) -> canonical short form.
STREET = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "str": "st", "saint": "st", "ste": "ste_",
    "sainte": "st", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave", "boulevard": "blvd",
    "blvd": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "court": "crt", "ct": "crt", "place": "pl", "pl": "pl", "plaza": "plz",
    "plz": "plz", "circle": "cir", "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq",
    "sq": "sq", "suite": "ste_", "unit": "unit", "apartment": "apt", "apt": "apt", "apts": "apt",
    "building": "bldg", "bldg": "bldg", "floor": "floor", "flr": "floor", "flat": "flat",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "heights": "hts", "hts": "hts", "township": "twp", "twp": "twp", "expressway": "expy",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "junction": "jct", "jct": "jct",
    # India
    "nagar": "nagar", "ngr": "nagar", "marg": "marg", "mg": "marg", "colony": "colony", "col": "colony",
    "sector": "sector", "sec": "sector", "phase": "phase", "ph": "phase", "industrial": "indl",
    "indl": "indl", "ind": "indl", "estate": "estate", "est": "estate", "complex": "complex",
    "cmplx": "complex", "chowk": "chowk", "bazar": "bazar", "bazaar": "bazar", "gali": "gali",
    "district": "dist", "dist": "dist", "distt": "dist", "taluk": "taluk", "taluka": "taluk",
    "tal": "taluk", "tq": "taluk", "village": "vill", "vill": "vill", "vpo": "vill", "post": "po",
    "po": "po", "cross": "cross", "crs": "cross", "main": "main", "layout": "layout", "extn": "extn",
    "extension": "extn", "opposite": "opp", "opp": "opp", "near": "near", "nr": "near",
    "behind": "behind", "bh": "behind", "beside": "beside", "besides": "beside", "adjacent": "adj",
    "adj": "adj", "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "ground": "gf", "gf": "gf",
    # France
    "rue": "rue", "r": "rue", "chemin": "chemin", "ch": "chemin", "chem": "chemin", "allee": "allee",
    "all": "allee", "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte", "quai": "quai",
    "qu": "quai", "faubourg": "fbg", "fbg": "fbg", "cours": "cours", "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat", "lieu": "lieu", "dit": "dit", "lotissement": "lot",
    "lot": "lot", "zone": "zone", "za": "za", "zi": "zi", "zac": "zac", "hameau": "ham",
    "ham": "ham", "passage": "pass", "pass": "pass", "cite": "cite", "sentier": "sent", "voie": "voie",
}
# Tokens that carry no discriminative information (dropped from "core" token sets).
ADDR_STOP = {"no", "nos", "number", "num", "n", "door", "the", "of", "and", "de", "du", "des", "la",
             "le", "les", "l", "d", "et", "a", "city", "town", "cdp", "na", "po", "box", "bis", "ter"}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d", "et", "a", "an",
             "for", "en", "au", "aux", "dba", "aka"}
STREET_TYPES = set(STREET.values()) | {"no"}
LANDMARK = {"near", "opp", "behind", "beside", "adj", "next", "front", "off"}

# Region / state abbreviations -> canonical token (hand-written, documented).
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    # romanised native-script spellings produced by translit_brahmic
    "maharashtr": "mh", "karnatak": "ka", "tamilnatu": "tn", "telangan": "tg",
    "pashchim bang": "wb", "pashchimabang": "wb", "pashchimbang": "wb", "keralam": "kl",
    "hariyana": "hr", "dilli": "dl", "andhr pradesh": "ap", "panjab": "pb",
}
# Full region names -> plain 2-letter codes (bare codes are already in that form).
_REGION_PHRASES = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))
_RE_REGION = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _REGION_PHRASES) + r")\b")
_REGION_MAP = dict(_REGION_PHRASES)

_WORD_ABBR = {  # generic name-word abbreviations
    "intl": "international", "int": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "srvcs": "services", "assoc": "associates",
    "assocs": "associates", "bros": "brothers", "mgmt": "management", "mgt": "management",
    "dept": "department", "natl": "national", "univ": "university", "hosp": "hospital",
    "ctr": "center", "centre": "center", "tech": "technologies", "technology": "technologies",
    "sys": "systems", "grp": "group", "hldgs": "holdings", "invest": "investments",
    "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises", "eng": "engineering",
    "engg": "engineering", "ind": "industries", "inds": "industries", "industry": "industries",
    "sol": "solutions", "soln": "solutions", "solution": "solutions", "st": "saint", "ste": "sainte",
    "dr": "doctor", "mt": "mount", "ft": "fort", "sr": "senior", "jr": "junior", "esq": "esquire",
    "phd": "phd", "md": "md", "dds": "dds",
}
_RE_DOMAIN = re.compile(r"(?:www\.)?([a-z0-9-]+)\.(com|net|org|in|co\.in|fr|biz|info|us|io|co)\b")
_RE_DIGITS = re.compile(r"\d+")
_RE_POSTAL = re.compile(r"(?<![\d/-])(\d{5,6}|\d{3} \d{3})(?![\d/-])")
_RE_NUMTOKEN = re.compile(r"[0-9]+[a-z]?(?:[/-][0-9a-z]+)*")


# =============================================================================== names
def _legal_phrases(s: str) -> str:
    for rx, rep in LEGAL_PHRASES:
        s = rx.sub(rep, s)
    return s


def normalize_name(raw: str, country: str = "") -> dict:
    """Return name fields: norm, core, legal, sorted, skel, nonlatin, domain."""
    low = raw.casefold()
    dm = _RE_DOMAIN.search(low)
    domain = dm.group(1).replace("-", "") if dm else ""
    s = fold(raw)
    s = _legal_phrases(s)
    toks = s.split()
    ctry = set(fold(country).split()) if country else set()
    legal, core = [], []
    for i, t in enumerate(toks):
        t = _WORD_ABBR.get(t, t) if t not in LEGAL else t
        if t in LEGAL and not (t in ("co", "sa", "pa", "ag", "pc", "lp") and len(toks) == 1):
            legal.append(LEGAL[t])
        elif t in ctry or t in ("www", "com", "http", "https"):
            continue
        else:
            core.append(t)
    norm = " ".join(_WORD_ABBR.get(t, t) for t in toks)
    core_nostop = [t for t in core if t not in NAME_STOP] or core
    return {
        "name_norm": norm,
        "name_core": " ".join(core_nostop),
        "name_legal": " ".join(sorted(set(legal))),
        "name_sorted": " ".join(sorted(core_nostop)),
        "name_skel": " ".join(sorted(skeleton(t) for t in core_nostop)),
        "name_nonlatin": has_nonlatin(raw),
        "name_domain": domain,
    }


_DEVOICE = str.maketrans("bfvgjqxzcsdm", "pppkkkkkkktn")  # Soundex-like consonant classes


@lru_cache(maxsize=200_000)
def skeleton(tok: str) -> str:
    """Consonant skeleton: digraph unification, Soundex-like classes, drop non-initial vowels, collapse repeats."""
    if tok.isdigit() or len(tok) <= 2:
        return tok
    t = tok
    for a, b in (("ph", "f"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"),
                 ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
                 ("gh", "g"), ("c", "k")):
        t = t.replace(a, b)
    t = t.translate(_DEVOICE)  # Indic scripts (e.g. Tamil) do not mark voicing; ச = ch/s; anusvara m/n
    head, rest = t[0], re.sub(r"[aeiouyh]", "", t[1:])
    out = head + rest
    return re.sub(r"(.)\1+", r"\1", out)


# =============================================================================== addresses
def normalize_address(raw: str) -> dict:
    """Return address fields: norm, core tokens, digits, postal, street number, landmark."""
    s = _RE_NUMALPHA.sub(" ", fold(raw))
    s = _RE_REGION.sub(lambda m: _REGION_MAP[m.group(1)], s)
    toks = s.split()
    canon = []
    for t in toks:
        canon.append(STREET.get(t, t))
    norm = " ".join(canon)
    digits = sorted({d.lstrip("0") or "0" for d in _RE_DIGITS.findall(s)})
    # postal-like: 5-6 digit groups (also "560 001", "75 001") that are not the first number
    raw_f = fold(raw)
    nums = list(_RE_NUMTOKEN.finditer(raw_f))
    postal = ""
    pm = list(_RE_POSTAL.finditer(raw_f))
    if pm:
        last = pm[-1]
        first_num_start = nums[0].start() if nums else -1
        if last.start() != first_num_start or len(pm) > 1:
            postal = last.group(1).replace(" ", "")
    street_num = ""
    street_word = ""
    for m in nums:
        tok = m.group(0)
        if tok.replace(" ", "") == postal:
            continue
        street_num = tok.lstrip("0") or "0"
        after = raw_f[m.end():].split()
        after = [STREET.get(a, a) for a in after]
        after = [a for a in after if a not in ADDR_STOP and a not in STREET_TYPES and not a[0].isdigit()]
        street_word = after[0] if after else ""
        break
    core = [t for t in canon if t not in ADDR_STOP]
    return {
        "addr_norm": norm,
        "addr_core": " ".join(core),
        "addr_sorted": " ".join(sorted(set(core))),
        "addr_digits": " ".join(digits),
        "addr_postal": postal,
        "addr_stnum": street_num,
        "addr_stword": street_word,
        "addr_landmark": any(t in LANDMARK for t in canon),
        "addr_nonlatin": has_nonlatin(raw),
    }


NAME_FIELDS = ["name_norm", "name_core", "name_legal", "name_sorted", "name_skel", "name_nonlatin", "name_domain"]
ADDR_FIELDS = ["addr_norm", "addr_core", "addr_sorted", "addr_digits", "addr_postal", "addr_stnum",
               "addr_stword", "addr_landmark", "addr_nonlatin"]


def normalize_records(names: list[str], addrs: list[str], countries: list[str]) -> dict[str, list]:
    """Normalise parallel lists (used inside multiprocessing workers)."""
    out: dict[str, list] = {k: [] for k in NAME_FIELDS + ADDR_FIELDS}
    for n, a, c in zip(names, addrs, countries):
        for k, v in normalize_name(n, c).items():
            out[k].append(v)
        for k, v in normalize_address(a).items():
            out[k].append(v)
    return out
