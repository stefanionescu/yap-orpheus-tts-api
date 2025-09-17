AUDIO_PRE  = "<custom_token_3><|begin_of_text|>"
AUDIO_POST = "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"

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
    return f"{AUDIO_PRE}{v}: {text}{AUDIO_POST}"


