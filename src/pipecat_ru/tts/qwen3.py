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

The ``qwen-tts`` package has no incremental-audio API yet, so each text chunk is
synthesized in one call. The Russian splitter keeps the first chunk of a reply short,
which is what keeps the time to first audio low.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language

from pipecat_ru.tts._base import RussianLocalTTSService

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

    def synthesize(self, text: str) -> tuple[Any, int]:
        """Synthesize one chunk with the configured voice."""
        model = self._qwen_model
        if self._clone:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=self._qwen_language,
                voice_clone_prompt=self._clone_prompt,
                **self._generate_kwargs,
            )
        else:
            extra: dict[str, Any] = {"instruct": self._instruct} if self._instruct else {}
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=self._qwen_language,
                speaker=self._speaker,
                **extra,
                **self._generate_kwargs,
            )
        return wavs[0], int(sr)
