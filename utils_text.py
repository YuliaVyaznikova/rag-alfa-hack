import re
from typing import List, Tuple

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


_whitespace_re = re.compile(r"\s+")
_word_re = re.compile(r"\b\w+\b", re.UNICODE)


def clean_text(raw: str) -> str:
    """Lightweight text cleanup: strip HTML (if bs4 available), collapse spaces."""
    if not raw:
        return ""
    txt = raw
    if BeautifulSoup is not None and ("<" in raw and ">" in raw):
        try:
            txt = BeautifulSoup(raw, "lxml").get_text(" ")
        except Exception:
            try:
                txt = BeautifulSoup(raw, "html.parser").get_text(" ")
            except Exception:
                txt = raw
    txt = txt.replace("\u00A0", " ")
    txt = _whitespace_re.sub(" ", txt).strip()
    return txt


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return _word_re.findall(text)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """Chunk text by characters with overlap. Returns list of (start, end, chunk)."""
    if not text:
        return []
    if chunk_size <= 0:
        return [(0, len(text), text)]
    res: List[Tuple[int, int, str]] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - max(0, overlap))
    while i < n:
        j = min(n, i + chunk_size)
        res.append((i, j, text[i:j]))
        i += step
    return res
