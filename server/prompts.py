ALIASES = {
    "female": "tara",
    "male": "zac",
}

def resolve_voice(v: str) -> str:
    if not v:
        return "tara"
    key = v.strip().lower()
    return ALIASES.get(key, key if key in ("tara", "zac") else "tara")

def build_prompt(text: str, voice: str = "tara") -> str:
    v = resolve_voice(voice)
    # Orpheus finetuned prompt format: "<voice>: <text>"
    return f"{v}: {text}"


