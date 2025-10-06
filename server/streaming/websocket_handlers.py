"""WebSocket message handling logic extracted from server.py for clean separation."""

import asyncio
import json
from typing import Optional

from server.config import settings
from server.prompts import resolve_voice


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
            
            # Text message with optional voice override
            text = (obj.get("text") or "").strip()
            voice = obj.get("voice")
            if text:
                # Validate voice parameter if provided
                if voice is not None:
                    try:
                        resolve_voice(str(voice))  # This will raise ValueError if invalid
                    except ValueError as e:
                        raise ValueError(f"Voice validation failed: {e}")
                return {"type": "text", "text": text, "voice": voice}
        
        return None


class ConnectionState:
    """Manages per-connection synthesis parameters."""
    
    def __init__(self):
        self.voice = settings.default_voice
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None
        self.repetition_penalty: Optional[float] = None
        self.max_tokens: Optional[int] = None
    
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
        
        for param, attr in [
            ("temperature", "temperature"),
            ("top_p", "top_p"), 
            ("repetition_penalty", "repetition_penalty"),
            ("max_tokens", "max_tokens")
        ]:
            if param in meta:
                try:
                    value = float(meta[param]) if param != "max_tokens" else int(meta[param])
                    setattr(self, attr, value)
                except Exception:
                    pass
    
    def get_sampling_kwargs(self) -> dict:
        """Build sampling parameters dict with fallback defaults."""
        return {
            "temperature": float(
                self.temperature if self.temperature is not None else settings.default_temperature
            ),
            "top_p": float(self.top_p if self.top_p is not None else settings.default_top_p),
            "repetition_penalty": float(
                self.repetition_penalty if self.repetition_penalty is not None else settings.default_repetition_penalty
            ),
            "max_tokens": int(self.max_tokens if self.max_tokens is not None else settings.orpheus_max_tokens),
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
