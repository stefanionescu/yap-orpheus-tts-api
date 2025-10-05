import re
from typing import List

_ABBR = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "st.", "no.",
    "inc.", "ltd.", "etc.", "e.g.", "i.e.", "al.", "fig.", "dept.", "est.",
    "col.", "gen.", "cmdr.", "u.s.", "u.k.", "u.n.", "a.m.", "p.m.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec."
}


def _split_sentences_fast(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    out: List[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?=(?:['\")\]]*\s|$))", s):
        end = m.end()
        seg = s[start:end].strip()
        if seg:
            last = seg.split()[-1].lower()
            last = re.sub(r"[^a-z.]+", "", last)
            if last in _ABBR:
                continue
            out.append(seg)
            start = end
    tail = s[start:].strip()
    if tail:
        out.append(tail)
    return out


def chunk_by_sentences(
    text: str
) -> list[str]:
    """
    Split text into sentences only (no word-based chunking).
    Production behavior: client sends sentence-by-sentence.
    """
    sents = _split_sentences_fast(text)
    return [sent.strip() for sent in sents if sent.strip()]


