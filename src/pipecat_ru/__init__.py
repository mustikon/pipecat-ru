"""Russian-language plugins for the Pipecat voice framework.

Light parts import eagerly; the model-backed services live in submodules so that
importing the package never pulls in torch:

- ``pipecat_ru.text``: normalizer, stress marks, sentence/clause splitter
- ``pipecat_ru.turns``: backchannel-aware user turn start strategy
- ``pipecat_ru.stt.gigaam``: GigaAM v3 speech-to-text
- ``pipecat_ru.tts.qwen3``: Qwen3-TTS (voice cloning)
- ``pipecat_ru.tts.silero``: Silero v5 TTS (CPU fallback; non-commercial weights)
"""

__version__ = "0.1.0"
