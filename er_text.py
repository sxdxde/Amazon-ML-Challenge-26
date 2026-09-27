"""Language-general text normalisation (no external data, no GPL: unicodedata only).

Pipeline per string: NFKD -> Brahmic-script transliteration (hand-written table; all nine
ISCII-aligned Indic blocks share one offset table) -> strip combining marks -> casefold ->
'&' -> 'and' -> punctuation/whitespace normalisation. Names then get their legal suffix split
off; addresses get abbreviation expansion and digit / postal / street-number / landmark extraction.
Nothing here depends on the country value except removing the record's own country string
from its name (data-driven, works for any unseen country).
"""
import re
import unicodedata

import pandas as pd
import pyarrow as pa
from joblib import Parallel, delayed

# ---------------------------------------------------------------- Brahmic transliteration
# Offsets inside a 128-codepoint Indic block (Devanagari layout; Bengali/Gurmukhi/Gujarati/Oriya/
# Tamil/Telugu/Kannada/Malayalam use the same offsets for the same sounds).
_BRAHMIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_IND_VOWEL = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
              0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
              0x60: "ri", 0x61: "li"}
_CONS = dict(zip(range(0x15, 0x3A), [
    "k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n",
    "t", "th", "d", "dh", "n", "n", "p", "ph", "b", "bh", "m", "y", "r", "r", "l", "l", "l",
    "v", "sh", "sh", "s", "h"]))
_CONS.update({0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y"})
_MATRA = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
          0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h"}  # candrabindu, anusvara, visarga
_BRAHMIC_RE = re.compile("[ऀ-ൿ]")


def _brahmic_off(ch):
    cp = ord(ch)
    if 0x0900 <= cp <= 0x0D7F:
        return cp - _BRAHMIC_BASES[(cp - 0x0900) // 0x80]
    return None


def translit_brahmic(s):
    """Rough phonetic Latin rendering, e.g. 'प्राइवेट लिमिटेड' -> 'praivet limited'."""
    if not _BRAHMIC_RE.search(s):
        return s
    out, pending_a = [], False
    chars = list(s)
    for i, ch in enumerate(chars):
        off = _brahmic_off(ch)
        if off is None:
            pending_a = False  # word-final schwa deleted
            out.append(ch)
            continue
        if off == _NUKTA:
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            # inherent 'a' only if another consonant follows (schwa deletion at word end /
            # before independent vowels, which is how English words are written in Devanagari)
            nxt = next((_brahmic_off(c) for c in chars[i + 1:] if _brahmic_off(c) != _NUKTA), None)
            pending_a = nxt in _CONS or nxt in _SIGNS
            continue
        if off in _MATRA:
            out.append(_MATRA[off]); pending_a = False; continue
        if off == _VIRAMA:
            pending_a = False; continue
        if pending_a:
            out.append("a"); pending_a = False
        if off in _SIGNS:
            out.append(_SIGNS[off])
        elif off in _IND_VOWEL:
            out.append(_IND_VOWEL[off])
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        else:
            out.append(" ")
    return "".join(out)


# ---------------------------------------------------------------- dictionaries (hand-written)
LEGAL = {
    "ltd": "ltd", "limited": "ltd", "limitd": "ltd", "limtd": "ltd", "limted": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "praivet": "pvt", "pravet": "pvt",
    "corp": "corp", "corporation": "corp", "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp", "elelpi": "llp", "lp": "lp", "pllc": "llc", "plc": "plc",
    "co": "co", "company": "co", "cie": "co", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "bld": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "trl": "trail", "ter": "terrace", "sq": "square", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "mg": "marg", "ngr": "nagar", "nr": "near",
    "opp": "opposite", "r": "rue", "ch": "chemin", "chem": "chemin", "pl": "place", "imp": "impasse",
    "all": "allee", "rte": "route", "fbg": "faubourg", "qu": "quai", "crs": "cours",
}
ADDR_DROP = {"no", "number", "n", "num", "unit", "hno", "h"}  # tokens with no identity value
LANDMARK = {"near", "opposite", "behind", "beside", "facing", "pres", "proche", "cote", "derriere"}
LANDMARK_BIGRAMS = ("next to", "en face", "a cote", "pres de")

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+")
_DOT_AFTER_LETTER = re.compile(r"(?<=[^\W\d_])\.")
_APOS_S = re.compile(r"['’]s\b")
_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_PIN_SPLIT = re.compile(r"\b(\d{3}) (\d{3})\b")
_DIGITS = re.compile(r"\d+")


def base_norm(s):
    """NFKD + transliterate + strip accents + casefold + & -> and + punctuation cleanup."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = translit_brahmic(s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = s.replace("&", " and ").replace("+", " and ")
    s = _APOS_S.sub("s", s)
    s = _DOT_AFTER_LETTER.sub("", s)  # l.l.c. -> llc, pvt. -> pvt, s.a.s -> sas
    s = s.replace("_", " ")
    s = _NON_WORD.sub(" ", s)
    s = _LETTER_DIGIT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_name(raw, country_tokens):
    """-> (name_n, name_core, legal, name_sorted)."""
    toks = base_norm(raw).split()
    name_n = " ".join(toks)
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in country_tokens] or toks
    return name_n, " ".join(core), " ".join(legal), " ".join(sorted(core))


def norm_addr(raw):
    """-> (addr_n, digits, postal, stnum, landmark)."""
    s = base_norm(raw)
    s = _PIN_SPLIT.sub(r"\1\2", s)
    landmark = int(any(b in s for b in LANDMARK_BIGRAMS))
    toks = []
    for t in s.split():
        t = ADDR_ABBR.get(t, t)
        if t in LANDMARK:
            landmark = 1
        if t in ADDR_DROP:
            continue
        toks.append(t)
    runs = [t for t in toks if t.isdigit()]
    postal = next((d for d in reversed(runs) if 5 <= len(d) <= 6), "")
    if not postal:
        postal = next((d for i, d in enumerate(reversed(runs)) if len(d) == 4 and i < len(runs) - 1), "")
    stnum = next((d.lstrip("0") or "0" for d in runs if d != postal and len(d) <= 5), "")
    return " ".join(toks), " ".join(sorted(set(runs))), postal, stnum, landmark


def _norm_chunk(names, addrs, countries):
    out = {k: [] for k in ("name_n", "name_core", "legal", "name_sorted",
                           "addr_n", "digits", "postal", "stnum", "landmark")}
    ctok_cache = {}
    for nm, ad, c in zip(names, addrs, countries):
        ct = ctok_cache.get(c)
        if ct is None:
            ct = ctok_cache[c] = set(base_norm(c).split())
        a, b, l, so = norm_name(nm, ct)
        ad_n, dg, pc, sn, lm = norm_addr(ad)
        out["name_n"].append(a); out["name_core"].append(b); out["legal"].append(l)
        out["name_sorted"].append(so); out["addr_n"].append(ad_n); out["digits"].append(dg)
        out["postal"].append(pc); out["stnum"].append(sn); out["landmark"].append(lm)
    out["landmark"] = pa.array(out["landmark"], type=pa.int8())
    return pa.table(out)  # compact to pickle back from the worker


def normalize_records(rec, n_jobs, chunk=100_000):
    """Adds normalised columns to the records table (parallel over row chunks, slices built lazily)."""
    spans = [(i, min(i + chunk, len(rec))) for i in range(0, len(rec), chunk)]
    cols = ("business_name", "business_address", "country")
    tabs = Parallel(n_jobs=n_jobs, batch_size=1, pre_dispatch="2*n_jobs")(
        delayed(_norm_chunk)(*(rec[c].iloc[a:b].tolist() for c in cols)) for a, b in spans)
    tab = pa.concat_tables(tabs)
    del tabs
    for k in tab.column_names:
        col = tab.column(k)
        rec[k] = col.to_numpy() if k == "landmark" else pd.arrays.ArrowExtensionArray(col)
    return rec


def str_take(rec, col, idx):
    """Gather strings for integer row indices without materialising the whole column in Python."""
    ca = rec[col].array.__arrow_array__()
    return ca.take(pa.array(idx)).to_pylist()


def acronym(core):
    return "".join(t[0] for t in core.split() if t)


def emb_text(s):
    """Text fed to the embedder: original script kept (bge-m3 is multilingual), junk collapsed."""
    s = _WS.sub(" ", s.replace("\t", " ")).strip(" -*<>@#|~_")
    return s if s else " "


if __name__ == "__main__":  # quick manual check (not run automatically)
    for x in ["प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड", "आदित्य प्रॉपर्टीज एलएलपी",
              "LLC Moncada Léarning Center", "Maure Hardin (L.L.C.) (L.L.C.)"]:
        print(x, "->", norm_name(x, set()))
    print(norm_addr("N° 14 R. DE BOULMGNE, TOURCOING"), norm_addr("Flat 12, Opp. SBI, Pune 411 001"))
