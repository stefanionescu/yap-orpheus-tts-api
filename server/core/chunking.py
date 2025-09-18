import os
import re
from typing import List

# Sentence-safe, word-based chunking (latency-first)
FIRST_CHUNK_WORDS = int(os.getenv("FIRST_CHUNK_WORDS", "40"))
NEXT_CHUNK_WORDS = int(os.getenv("NEXT_CHUNK_WORDS", "140"))
MIN_TAIL_WORDS = int(os.getenv("MIN_TAIL_WORDS", "12"))

_ABBR = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "st.", "no.",
    "inc.", "ltd.", "etc.", "e.g.", "i.e.", "al.", "fig.", "dept.", "est.",
    "col.", "gen.", "cmdr.", "u.s.", "u.k.", "u.n.", "a.m.", "p.m.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec."
}

_WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)


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


def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s))


def chunk_by_words(
    text: str,
    first_chunk_words: int = FIRST_CHUNK_WORDS,
    next_chunk_words: int = NEXT_CHUNK_WORDS,
    min_tail_words: int = MIN_TAIL_WORDS,
) -> list[str]:
    sents = _split_sentences_fast(text)
    if not sents:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_wc = 0
    target = max(1, first_chunk_words)

    for sent in sents:
        wc = _word_count(sent)

        if not buf and wc > target:
            chunks.append(sent.strip())
            target = max(1, next_chunk_words)
            continue

        if buf and (buf_wc + wc) > target:
            chunks.append(" ".join(buf).strip())
            buf, buf_wc = [sent], wc
            target = max(1, next_chunk_words)
        else:
            buf.append(sent)
            buf_wc += wc

    if buf:
        chunks.append(" ".join(buf).strip())

    if min_tail_words > 0 and len(chunks) >= 2 and _word_count(chunks[-1]) < min_tail_words:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()

    return chunks


