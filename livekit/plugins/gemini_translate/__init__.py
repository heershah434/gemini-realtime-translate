"""
Gemini Live Translate integration for the LiveKit realtime translation pipeline.

Exposes a `GeminiTranslationModel` / `GeminiTranslationSession` pair that wraps
Google's dedicated speech-to-speech translation model (`gemini-3.5-live-translate-preview`)
on the Gemini Live API. It subclasses `livekit-plugins-google`'s realtime
`RealtimeModel` / `RealtimeSession` so it can be used interchangeably with the
OpenAI realtime / translate and Azure Speech translate providers.

Wire it up through `get_llm` with metadata `llm.type = "GOOGLE"` and
`llm.is_translation = true`.
"""

from .translation_model import GeminiTranslationModel, GeminiTranslationSession
from .version import __version__

__all__ = [
    "GeminiTranslationModel",
    "GeminiTranslationSession",
    "__version__",
]
