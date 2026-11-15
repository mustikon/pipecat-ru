"""GigaAM v3 speech-to-text for Pipecat.

`GigaAM <https://github.com/salute-developers/GigaAM>`_ (MIT, SberDevices) is the
strongest open Russian ASR family. The ``v3_e2e_*`` models return final text with
punctuation and normalization, and unlike Whisper they do not hallucinate on silence.

Install::

    pip install "pipecat-ru[gigaam]"
    pip install "gigaam @ git+https://github.com/salute-developers/GigaAM"

The service is segmented: Pipecat's VAD cuts the user's speech into utterances and
each utterance is transcribed in one call, on the GPU if there is one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from pipecat_ru._audio import FloatArray, pcm16_to_float, split_long

GIGAAM_SAMPLE_RATE = 16000
DEFAULT_MODEL = "v3_e2e_rnnt"
# GigaAM's .transcribe() refuses audio over 25 s; stay below with a margin.
MAX_SEGMENT_SECS = 24.0


@dataclass
class GigaAMSTTSettings(STTSettings):
    """Settings for :class:`GigaAMSTTService`.

    Parameters:
        model: GigaAM model name: ``v3_e2e_rnnt`` (default, best quality),
            ``v3_e2e_ctc`` (faster), or ``v3_rnnt``/``v3_ctc`` (no punctuation).
        language: Always Russian.
    """


def load_gigaam_model(
    model_name: str = DEFAULT_MODEL,
    *,
    device: str | None = None,
    fp16_encoder: bool = True,
    use_flash: bool = False,
    download_root: str | None = None,
) -> Any:
    """Load a GigaAM model (downloads it on first use)."""
    try:
        import gigaam
    except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
        raise ImportError(
            "GigaAM is not installed. Run:\n"
            '  pip install "pipecat-ru[gigaam]" '
            '"gigaam @ git+https://github.com/salute-developers/GigaAM"'
        ) from e
    kwargs: dict[str, Any] = {"fp16_encoder": fp16_encoder, "use_flash": use_flash}
    if device is not None:
        kwargs["device"] = device
    if download_root is not None:
        kwargs["download_root"] = download_root
    return gigaam.load_model(model_name, **kwargs)


def _result_text(result: Any) -> str:
    """GigaAM returns a TranscriptionResult (``.text``) in v3 and a str before."""
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return str(text or "").strip()


class GigaAMTranscriber:
    """Synchronous GigaAM wrapper: float32 16 kHz audio in, text out.

    Feeds audio to the model as a tensor (no temporary file, no ffmpeg). If the
    model's internals differ from what this expects, it falls back to GigaAM's
    public ``transcribe(path)`` API with a temporary WAV file and stays there.
    """

    def __init__(self, model: Any, *, max_segment_secs: float = MAX_SEGMENT_SECS) -> None:
        """Wrap a loaded GigaAM model."""
        self.model = model
        self.max_segment_secs = max_segment_secs
        self.fast_path = all(hasattr(model, a) for a in ("forward", "_decode", "_device"))

    def transcribe(self, samples: FloatArray) -> str:
        """Transcribe mono float32 audio at 16 kHz; long audio is cut into pieces."""
        pieces = split_long(samples, GIGAAM_SAMPLE_RATE, self.max_segment_secs)
        texts = [self._transcribe_piece(p) for p in pieces if len(p)]
        return " ".join(t for t in texts if t)

    def _transcribe_piece(self, samples: FloatArray) -> str:
        if self.fast_path:
            try:
                return self._transcribe_tensor(samples)
            except (ImportError, AttributeError, TypeError, ValueError, IndexError) as e:
                logger.warning(f"GigaAM tensor input failed ({e!r}); using transcribe(path)")
                self.fast_path = False
        return self._transcribe_file(samples)

    def _transcribe_tensor(self, samples: FloatArray) -> str:
        import torch

        model = self.model
        with torch.inference_mode():
            wav = torch.from_numpy(samples).to(model._device).to(model._dtype).unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=model._device)
            encoded, encoded_len = model.forward(wav, length)
            decoded = model._decode(encoded, encoded_len, length, False)[0]
        text = decoded[0] if isinstance(decoded, tuple | list) else decoded
        return _result_text(text)

    def _transcribe_file(self, samples: FloatArray) -> str:
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="pipecat-ru-")
        os.close(fd)
        try:
            sf.write(path, samples, GIGAAM_SAMPLE_RATE, subtype="PCM_16")
            return _result_text(self.model.transcribe(path))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)


class GigaAMSTTService(SegmentedSTTService):
    """Pipecat STT service running GigaAM v3 locally.

    Args:
        model: An already loaded GigaAM model (share one between services or
            inject a test double). When None, the model is loaded in a worker
            thread when the pipeline starts.
        device: ``"cuda"``, ``"cpu"`` or None for GigaAM's own choice.
        fp16_encoder: Run the encoder in fp16 on the GPU.
        use_flash: Use flash attention (needs ``flash-attn``).
        download_root: Where GigaAM caches checkpoints.
        max_segment_secs: Utterances longer than this are split at quiet points.
        settings: Runtime settings (``model``).
        **kwargs: Passed to ``SegmentedSTTService`` (e.g. ``trailing_silence_secs``).
    """

    Settings = GigaAMSTTSettings
    _settings: GigaAMSTTSettings

    def __init__(
        self,
        *,
        model: Any = None,
        device: str | None = None,
        fp16_encoder: bool = True,
        use_flash: bool = False,
        download_root: str | None = None,
        max_segment_secs: float = MAX_SEGMENT_SECS,
        settings: GigaAMSTTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the service; see the class docstring for the arguments."""
        default_settings = self.Settings(model=DEFAULT_MODEL, language=Language.RU)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(settings=default_settings, **kwargs)
        self._gigaam_device = device
        self._fp16_encoder = fp16_encoder
        self._use_flash = use_flash
        self._download_root = download_root
        self._max_segment_secs = max_segment_secs
        self._transcriber: GigaAMTranscriber | None = (
            GigaAMTranscriber(model, max_segment_secs=max_segment_secs)
            if model is not None
            else None
        )

    @property
    def wants_wav_segments(self) -> bool:
        """Receive raw 16-bit PCM, not a WAV container."""
        return False

    @property
    def segment_sample_rate(self) -> int:
        """GigaAM reads 16 kHz audio."""
        return GIGAAM_SAMPLE_RATE

    def can_generate_metrics(self) -> bool:
        """This service reports processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """GigaAM is Russian-only."""
        return "ru"

    async def start(self, frame: StartFrame) -> None:
        """Load the model (if not injected) before audio arrives."""
        await super().start(frame)
        await self.ensure_loaded()

    async def ensure_loaded(self) -> GigaAMTranscriber:
        """Load the model now if needed; safe to call more than once."""
        if self._transcriber is None:
            model_name = str(self._settings.model or DEFAULT_MODEL)
            logger.info(f"{self}: loading GigaAM {model_name}...")
            t0 = time.perf_counter()
            model = await asyncio.to_thread(
                load_gigaam_model,
                model_name,
                device=self._gigaam_device,
                fp16_encoder=self._fp16_encoder,
                use_flash=self._use_flash,
                download_root=self._download_root,
            )
            self._transcriber = GigaAMTranscriber(model, max_segment_secs=self._max_segment_secs)
            logger.info(f"{self}: GigaAM loaded in {time.perf_counter() - t0:.1f}s")
        return self._transcriber

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ) -> None:
        """Tracing hook (the decorator records the transcript)."""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one VAD segment (16-bit PCM, mono, 16 kHz)."""
        try:
            transcriber = await self.ensure_loaded()
            await self.start_processing_metrics()
            text = await asyncio.to_thread(transcriber.transcribe, pcm16_to_float(audio))
            await self.stop_processing_metrics()
        except Exception as e:
            logger.exception(f"{self}: GigaAM transcription failed")
            yield ErrorFrame(error=f"GigaAM transcription failed: {e}")
            return
        if text:
            logger.debug(f"{self}: transcript [{text}]")
            await self._handle_transcription(text, True, Language.RU)
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.RU)
