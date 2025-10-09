"""WebSocket message handling logic extracted from server.py for clean separation."""

import asyncio
import json
from typing import Optional

from server.config import settings
from server.voices import resolve_voice, get_voice_defaults


class MessageParser:
    """Handles parsing and routing of WebSocket messages."""
    
    @staticmethod
    def parse_message(msg: str) -> Optional[dict]:
        """Parse incoming WebSocket message into structured format."""
        msg = msg.strip()
        
        # Check for end sentinel
        if msg == settings.ws_end_sentinel:
            return {"type": "end"}
        
        # Try JSON parse
        try:
            obj = json.loads(msg)
        except Exception:
            # Plain text fallback
            if msg:
                return {"type": "text", "text": msg}
            return None
        
        if isinstance(obj, dict):
            if obj.get("end") is True:
                return {"type": "end"}
            
            # Metadata-only message
            if ("text" not in obj) and (any(k in obj for k in settings.ws_meta_keys)):
                return {"type": "meta", "meta": obj}
            
            # Text message with optional voice and generation parameter overrides
            text = (obj.get("text") or "").strip()
            voice = obj.get("voice")
            trim_silence = obj.get("trim_silence")
            temperature = obj.get("temperature")
            top_p = obj.get("top_p")
            repetition_penalty = obj.get("repetition_penalty")
            
            if text:
                # Validate voice parameter if provided
                if voice is not None:
                    try:
                        resolve_voice(str(voice))  # This will raise ValueError if invalid
                    except ValueError as e:
                        raise ValueError(f"Voice validation failed: {e}")
                
                # Validate generation parameters if provided
                if temperature is not None:
                    try:
                        temp_val = float(temperature)
                        if not (0.3 <= temp_val <= 0.9):
                            raise ValueError(f"Temperature must be between 0.3 and 0.9, got {temp_val}")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid temperature parameter: {e}")
                
                if top_p is not None:
                    try:
                        top_p_val = float(top_p)
                        if not (0.7 <= top_p_val <= 1.0):
                            raise ValueError(f"top_p must be between 0.7 and 1.0, got {top_p_val}")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid top_p parameter: {e}")
                
                if repetition_penalty is not None:
                    try:
                        rep_val = float(repetition_penalty)
                        if not (1.1 <= rep_val <= 1.9):
                            raise ValueError(f"repetition_penalty must be between 1.1 and 1.9, got {rep_val}")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid repetition_penalty parameter: {e}")
                
                # Build message with all optional parameters
                msg = {"type": "text", "text": text, "voice": voice}
                if trim_silence is not None:
                    msg["trim_silence"] = trim_silence
                if temperature is not None:
                    msg["temperature"] = temperature
                if top_p is not None:
                    msg["top_p"] = top_p
                if repetition_penalty is not None:
                    msg["repetition_penalty"] = repetition_penalty
                return msg
        
        return None


class ConnectionState:
    """Manages per-connection synthesis parameters."""
    
    def __init__(self):
        # Voice must be explicitly provided by the client via meta or per-message override
        self.voice = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None
        self.repetition_penalty: Optional[float] = None
        # Default to global setting; clients can override per-connection or per-text
        self.trim_silence: bool = bool(settings.trim_leading_silence)
    
    def update_from_meta(self, meta: dict) -> None:
        """Update connection state from metadata message."""
        if "voice" in meta and meta["voice"]:
            try:
                # Validate voice parameter - only 'female' and 'male' allowed
                voice_str = str(meta["voice"])
                resolve_voice(voice_str)  # This will raise ValueError if invalid
                self.voice = voice_str
            except ValueError as e:
                raise ValueError(f"Voice validation failed: {e}")
        
        # Validate and set generation parameters with proper ranges
        if "temperature" in meta:
            try:
                value = float(meta["temperature"])
                if 0.3 <= value <= 0.9:
                    self.temperature = value
                else:
                    raise ValueError(f"Temperature must be between 0.3 and 0.9, got {value}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid temperature parameter: {e}")
        
        if "top_p" in meta:
            try:
                value = float(meta["top_p"])
                if 0.7 <= value <= 1.0:
                    self.top_p = value
                else:
                    raise ValueError(f"top_p must be between 0.7 and 1.0, got {value}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid top_p parameter: {e}")
        
        if "repetition_penalty" in meta:
            try:
                value = float(meta["repetition_penalty"])
                if 1.1 <= value <= 1.9:
                    self.repetition_penalty = value
                else:
                    raise ValueError(f"repetition_penalty must be between 1.1 and 1.9, got {value}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid repetition_penalty parameter: {e}")
        # Boolean parsing for trim_silence
        if "trim_silence" in meta:
            val = meta["trim_silence"]
            try:
                if isinstance(val, bool):
                    self.trim_silence = val
                elif isinstance(val, str):
                    self.trim_silence = val.strip().lower() in {"1", "true", "yes", "y", "on"}
                else:
                    self.trim_silence = bool(int(val))
            except Exception:
                # Ignore invalid boolean; keep previous setting
                pass
    
    def get_sampling_kwargs(self) -> dict:
        """Build sampling parameters dict with voice-specific fallback defaults."""
        # Voice must have been set by now via meta or message override
        if not self.voice:
            raise ValueError("Voice not set; client must provide 'voice' in metadata or per message.")
        voice_defaults = get_voice_defaults(self.voice)
        
        return {
            "temperature": float(
                self.temperature if self.temperature is not None else voice_defaults["temperature"]
            ),
            "top_p": float(
                self.top_p if self.top_p is not None else voice_defaults["top_p"]
            ),
            "repetition_penalty": float(
                self.repetition_penalty if self.repetition_penalty is not None else voice_defaults["repetition_penalty"]
            ),
            # Server-enforced output length; not client-overridable
            "max_tokens": int(settings.orpheus_max_tokens),
            "stop_token_ids": list(settings.server_stop_token_ids),
        }


async def message_receiver(ws, queue: asyncio.Queue) -> None:
    """Receive and parse WebSocket messages, putting structured results in queue."""
    parser = MessageParser()
    
    while True:
        try:
            msg = await ws.receive_text()
            parsed = parser.parse_message(msg)
            
            if parsed is None:
                continue
                
            await queue.put(parsed)
            
            if parsed["type"] == "end":
                break
                
        except ValueError:
            # Voice validation error - send end signal to close connection
            await queue.put({"type": "end", "error": "invalid_voice"})
            break
        except Exception:
            await queue.put({"type": "end"})
            break
