from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
ASCII_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"inc|incorporated|corp|corporation|ltd|limited|llc|llp|plc|gmbh|ag|bv|sarl|srl|sa|sas|"
    r"pvt|private|co|company"
    r")\b",
    flags=re.IGNORECASE,
)
ADDRESS_WORDS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def unicode_clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold().replace("&", " and ")
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def ascii_fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", _text(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = ASCII_PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def compact(value: str) -> str:
    return SPACE_RE.sub("", value or "")


def suffix_normalized(value: object) -> str:
    text = unicode_clean(value)
    text = LEGAL_SUFFIX_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def name_tokens(value: object) -> tuple[str, ...]:
    return tuple(t for t in unicode_clean(value).split() if len(t) > 1)


def clean_address(value: object) -> str:
    tokens = []
    for token in unicode_clean(value).split():
        tokens.append(ADDRESS_WORDS.get(token, token))
    return " ".join(tokens)


def numeric_tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", _text(value)))


def postal_like_tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"\b[a-zA-Z]?\d[a-zA-Z0-9 -]{2,10}\d\b", _text(value)))

