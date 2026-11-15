"""A Russian voice assistant on one PC: microphone -> GigaAM -> local LLM -> Qwen3-TTS.

    uv pip install "pipecat-ru[gigaam,qwen,silero]" "pipecat-ai[local]"
    uv pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
    python examples/local_bot.py

The LLM is any OpenAI-compatible server (vLLM, llama.cpp, LM Studio, Ollama):
set LLM_BASE_URL and LLM_MODEL. Put a 5-15 s recording of the target voice in
VOICE_WAV and its exact transcript in VOICE_TEXT. Without them a preset
Qwen3-TTS speaker is used.

Silero (CPU) stands by: if Qwen3-TTS keeps failing (for example the GPU runs out of
memory), Pipecat's failover switcher moves speech to Silero. Silero's weights are
non-commercial (CC BY-NC 4.0).
"""

from __future__ import annotations

import asyncio
import os

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from pipecat_ru.stt.gigaam import GigaAMSTTService
from pipecat_ru.text import RussianTextNormalizer
from pipecat_ru.tts.qwen3 import Qwen3TTSService
from pipecat_ru.tts.silero import SileroTTSService
from pipecat_ru.turns import russian_user_turn_strategies

SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Отвечай по-русски, коротко и разговорно, без списков, "
    "markdown и эмодзи: твой ответ будет озвучен."
)

VAD_STOP_SECS = 0.2  # smart-turn v3.2 is tuned for 0.2 s


async def main() -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = GigaAMSTTService(device="cuda")

    llm = OpenAILLMService(
        api_key=os.environ.get("LLM_API_KEY", "local"),
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        settings=OpenAILLMService.Settings(model=os.environ.get("LLM_MODEL", "local-model")),
    )

    # Names you use often, spelled the way they should sound.
    normalizer = RussianTextNormalizer(lexicon={"Мустикон": "Мустик+он"})
    voice_wav, voice_text = os.environ.get("VOICE_WAV"), os.environ.get("VOICE_TEXT")
    if voice_wav:
        qwen = Qwen3TTSService(ref_audio=voice_wav, ref_text=voice_text, normalizer=normalizer)
    else:
        qwen = Qwen3TTSService(speaker="Ryan", normalizer=normalizer)
    silero = SileroTTSService(speaker="xenia", normalizer=normalizer)
    tts = ServiceSwitcher(services=[qwen, silero], strategy_type=ServiceSwitcherStrategyFailover)

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
            # "ага"/"угу" while the bot talks no longer interrupt it; smart-turn v3.2
            # (Russian) decides when the user has finished.
            user_turn_strategies=russian_user_turn_strategies(vad_stop_secs=VAD_STOP_SECS),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
