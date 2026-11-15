"""Qwen3-TTS for Pipecat, with voice cloning from a Russian reference clip.

`Qwen3-TTS <https://github.com/QwenLM/Qwen3-TTS>`_ (Apache-2.0) speaks Russian and
clones a voice from a few seconds of reference audio. Install::

    pip install "pipecat-ru[qwen]"
    # optional, faster on RTX GPUs: pip install flash-attn --no-build-isolation

Two modes:

* **Voice clone** (``Qwen/Qwen3-TTS-12Hz-1.7B-Base`` or ``-0.6B-Base``): pass
  ``ref_audio`` (a WAV path or ``(samples, sample_rate)``) and ``ref_text`` (what is
  said in it). The clone prompt is built once at startup and reused.
* **Preset speaker** (``...-CustomVoice`` models): pass ``speaker`` (and optionally
  ``instruct``).

**Fast streaming path** (default on CUDA). ``qwen-tts`` generates with Hugging Face
``generate()``, which is launch-bound on a fast GPU and slower than real time (RTF about
1.4 on an RTX 5090). The service therefore runs generation through
:class:`~pipecat_ru.tts._qwen3_fast.Qwen3FastGenerator`: a static KV cache, one CUDA
graph per 80 ms codec frame, and audio decoded as frames come out. On an RTX 5090 this
gives first audio in under 100 ms and RTF about 0.25 with the 1.7B models. Pass
``fast=False`` to use the plain ``qwen-tts`` calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from loguru import logger
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language

from pipecat_ru.tts._base import RussianLocalTTSService
from pipecat_ru.tts._qwen3_fast import Qwen3FastGenerator

DEFAULT_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def load_qwen3_tts(
    model_name: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Any:
    """Load a Qwen3-TTS model (downloads it from Hugging Face on first use)."""
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
        raise ImportError('Qwen3-TTS is not installed: pip install "pipecat-ru[qwen]"') from e
    kwargs: dict[str, Any] = {"device_map": device, "dtype": getattr(torch, dtype)}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return Qwen3TTSModel.from_pretrained(model_name, **kwargs)


def supports_fast_path(model: Any) -> bool:
    """True for a loaded 12 Hz ``qwen_tts.Qwen3TTSModel`` on a CUDA device."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "talker"):
        return False
    decoder = getattr(getattr(inner.speech_tokenizer, "model", None), "decoder", None)
    if not hasattr(decoder, "total_upsample"):
        return False
    try:
        return bool(next(inner.talker.parameters()).device.type == "cuda")
    except (StopIteration, AttributeError):
        return False


class Qwen3TTSService(RussianLocalTTSService):
    """Pipecat TTS service running Qwen3-TTS locally.

    Args:
        ref_audio: Reference clip for voice cloning: a path/URL, or ``(samples, sr)``.
        ref_text: Transcript of ``ref_audio`` (needed unless ``x_vector_only``).
        x_vector_only: Clone from the speaker embedding only (no ``ref_text``;
            lower quality).
        speaker: Preset speaker for CustomVoice models (``"Ryan"``, ``"Serena"``...).
        instruct: Style instruction for CustomVoice models.
        model: An already loaded ``Qwen3TTSModel`` (or a test double).
        model_name: Hugging Face id or local path. Defaults to the 1.7B Base model
            when cloning and the 1.7B CustomVoice model with a preset speaker.
        device: Torch device map, e.g. ``"cuda:0"``.
        dtype: ``"bfloat16"`` (default) or ``"float16"``.
        attn_implementation: ``"flash_attention_2"`` if flash-attn is installed.
        language: Language name passed to Qwen3-TTS.
        generate_kwargs: Extra generation arguments (``max_new_tokens``, ``top_p``...).
        fast: Use the CUDA-graph streaming generator. ``None`` (default) turns it on
            for a 12 Hz model on a CUDA device.
        first_chunk_frames: Fast path: 80 ms codec frames in the first audio chunk.
        chunk_frames: Fast path: frames in each later chunk.
        max_seq_len: Fast path: talker KV-cache length (prompt plus generated frames,
            12.5 frames per second of audio).
        **kwargs: Passed to :class:`~pipecat_ru.tts._base.RussianLocalTTSService`
            (``normalizer``, ``splitter``) and Pipecat's ``TTSService``.
    """

    def __init__(
        self,
        *,
        ref_audio: Any = None,
        ref_text: str | None = None,
        x_vector_only: bool = False,
        speaker: str | None = None,
        instruct: str | None = None,
        model: Any = None,
        model_name: str | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        language: str = "Russian",
        generate_kwargs: dict[str, Any] | None = None,
        fast: bool | None = None,
        first_chunk_frames: int = 3,
        chunk_frames: int = 8,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Create the service; see the class docstring for the arguments."""
        if ref_audio is None and speaker is None:
            raise ValueError("Qwen3TTSService needs ref_audio (voice clone) or speaker")
        if ref_audio is not None and ref_text is None and not x_vector_only:
            raise ValueError("Voice cloning needs ref_text (or x_vector_only=True)")
        self._clone = ref_audio is not None
        default_model = DEFAULT_CLONE_MODEL if self._clone else DEFAULT_CUSTOM_VOICE_MODEL
        settings = TTSSettings(
            model=model_name or default_model,
            voice=speaker or "clone",
            language=Language.RU,
        )
        kwargs.setdefault("settings", settings)
        super().__init__(**kwargs)
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._x_vector_only = x_vector_only
        self._speaker = speaker
        self._instruct = instruct
        self._qwen_model = model
        self._model_name = str(settings.model)
        self._qwen_device = device
        self._dtype = dtype
        self._attn = attn_implementation
        self._qwen_language = language
        self._generate_kwargs = generate_kwargs or {}
        self._clone_prompt: Any = None
        self._fast_option = fast
        self._first_chunk_frames = first_chunk_frames
        self._chunk_frames = chunk_frames
        self._max_seq_len = max_seq_len
        self._fast: Qwen3FastGenerator | None = None

    def language_to_service_language(self, language: Language) -> str | None:
        """Qwen3-TTS takes language names; this service is set up for Russian."""
        return self._qwen_language if hasattr(self, "_qwen_language") else "Russian"

    def load_model(self) -> None:
        """Load the model and build the voice clone prompt once."""
        if self._qwen_model is None:
            logger.info(f"{self}: loading {self._model_name} on {self._qwen_device}...")
            self._qwen_model = load_qwen3_tts(
                self._model_name,
                device=self._qwen_device,
                dtype=self._dtype,
                attn_implementation=self._attn,
            )
        if self._clone and self._clone_prompt is None:
            self._clone_prompt = self._qwen_model.create_voice_clone_prompt(
                ref_audio=self._ref_audio,
                ref_text=self._ref_text,
                x_vector_only_mode=self._x_vector_only,
            )
        use_fast = self._fast_option
        if use_fast is None:
            use_fast = supports_fast_path(self._qwen_model)
        if use_fast and self._fast is None:
            fast = Qwen3FastGenerator(self._qwen_model, max_seq_len=self._max_seq_len)
            fast.warmup(
                self._generate_call("Привет, сейчас всё расскажу."),
                ref_codes=self._ref_codes(),
                first_chunk_frames=self._first_chunk_frames,
                chunk_frames=self._chunk_frames,
            )
            self._fast = fast
            logger.info(f"{self}: fast streaming generation on")

    def _generate_call(self, text: str) -> Callable[[], Any]:
        """A zero-argument ``qwen-tts`` call that synthesizes ``text``."""
        model = self._qwen_model
        if self._clone:
            return lambda: model.generate_voice_clone(
                text=text,
                language=self._qwen_language,
                voice_clone_prompt=self._clone_prompt,
                **self._generate_kwargs,
            )
        extra: dict[str, Any] = {"instruct": self._instruct} if self._instruct else {}
        return lambda: model.generate_custom_voice(
            text=text,
            language=self._qwen_language,
            speaker=self._speaker,
            **extra,
            **self._generate_kwargs,
        )

    def _ref_codes(self) -> Any:
        """Codec frames of the reference clip (ICL voice clone), else None."""
        if not self._clone or not self._clone_prompt:
            return None
        item = self._clone_prompt[0]
        return getattr(item, "ref_code", None) if getattr(item, "icl_mode", False) else None

    def synthesize_stream(self, text: str) -> Iterator[tuple[Any, int]]:
        """Stream audio chunks (fast path) or yield the whole chunk at once."""
        fast = self._fast
        if fast is None:
            yield self.synthesize(text)
            return
        for chunk in fast.stream(
            self._generate_call(text),
            ref_codes=self._ref_codes(),
            first_chunk_frames=self._first_chunk_frames,
            chunk_frames=self._chunk_frames,
        ):
            yield chunk, fast.sample_rate

    def synthesize(self, text: str) -> tuple[Any, int]:
        """Synthesize one chunk with the configured voice."""
        fast = self._fast
        if fast is not None:
            chunks = [c for c, _ in self.synthesize_stream(text)]
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            return audio, fast.sample_rate
        wavs, sr = self._generate_call(text)()
        return wavs[0], int(sr)
