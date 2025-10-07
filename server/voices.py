"""Voice configuration and parameter management."""

# Voice aliases mapping external names to internal model names
VOICE_ALIASES = {
    "female": "tara",
    "male": "zac",
}


def resolve_voice(v: str) -> str:
    """
    Resolve voice parameter to internal voice name.
    Only accepts 'female' and 'male' as valid inputs.
    
    Args:
        v: Voice parameter string
        
    Returns:
        Internal voice name ("tara" or "zac")
        
    Raises:
        ValueError: If voice parameter is not 'female' or 'male'
    """
    if not v:
        return "tara"  # Default to female voice
    
    key = v.strip().lower()
    
    # Only allow 'female' and 'male' as valid voice parameters
    if key not in VOICE_ALIASES:
        raise ValueError(f"Invalid voice parameter '{v}'. Only 'female' and 'male' are supported.")
    
    return VOICE_ALIASES[key]


def get_voice_defaults(voice: str) -> dict:
    """
    Get voice-specific default sampling parameters.
    
    Based on optimal settings:
    - Female (Tara): temperature=0.45, top_p=0.95, repetition_penalty=1.25
    - Male (Zac): temperature=0.55, top_p=0.95, repetition_penalty=1.15
    
    Args:
        voice: Voice parameter ('female' or 'male')
        
    Returns:
        Dict with default temperature, top_p, repetition_penalty for the voice
    """
    resolved = resolve_voice(voice) if voice else "tara"
    
    if resolved == "zac":  # Male voice
        return {
            "temperature": 0.55,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
        }
    else:  # Female voice (tara) - default
        return {
            "temperature": 0.45,
            "top_p": 0.95,
            "repetition_penalty": 1.25,
        }


def get_available_voices() -> list[str]:
    """Get list of available voice parameters."""
    return list(VOICE_ALIASES.keys())


def get_voice_info() -> dict:
    """Get comprehensive voice configuration info."""
    return {
        "available_voices": get_available_voices(),
        "aliases": VOICE_ALIASES,
        "defaults": {
            voice: get_voice_defaults(voice) 
            for voice in get_available_voices()
        }
    }
