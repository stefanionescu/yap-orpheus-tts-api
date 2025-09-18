import re

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")


def split_custom_tokens(s: str) -> list[int]:
    return [int(x) for x in _TOKEN_RE.findall(s) if x != "0"]


def turn_token_into_id(token_number: int, index: int) -> int:
    # Baseten’s exact rule
    return token_number - 10 - ((index % 7) * 4096)


