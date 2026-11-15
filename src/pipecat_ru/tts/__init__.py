"""Local Russian TTS services.

Import the service you need from its module so only that model's dependencies load::

    from pipecat_ru.tts.qwen3 import Qwen3TTSService
    from pipecat_ru.tts.silero import SileroTTSService
"""

from pipecat_ru.tts._base import RussianLocalTTSService

__all__ = ["RussianLocalTTSService"]
