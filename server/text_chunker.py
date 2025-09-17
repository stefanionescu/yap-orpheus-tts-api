import re
from typing import List

def chunk_text(text: str, target_chars: int = 500) -> List[str]:
    """
    Split long text for ≥60s utterances while avoiding hard truncation mid-sentence.
    """
    if len(text) <= target_chars:
        return [text]

    sentences = re.split(r'(?<=[\.!\?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > target_chars and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks


