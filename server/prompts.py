AUDIO_PRE  = "<custom_token_3><|begin_of_text|>"
AUDIO_POST = "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"

def build_prompt(text: str, voice: str = "tara") -> str:
    # Orpheus: "<voice>: <text>" framed with audio control tokens
    return f"{AUDIO_PRE}{voice}: {text}{AUDIO_POST}"


