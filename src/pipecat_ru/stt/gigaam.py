"""GigaAM v3 speech-to-text for Pipecat.

`GigaAM <https://github.com/salute-developers/GigaAM>`_ (MIT, SberDevices) is the
strongest open Russian ASR family. The ``v3_e2e_*`` models return final text with
punctuation and normalization, and unlike Whisper they do not hallucinate on silence.

Install::

    pip install "pipecat-ru[gigaam]"
    pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"

The service is segmented: Pipecat's VAD cuts the user's speech into utterances and
each utterance is transcribed in one call, on the GPU if there is one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
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


# The official Hugging Face mirror of the checkpoints, used when SberDevices' CDN is
# unreachable. Revisions are pinned to commits so the downloaded weights cannot change.
HF_REPO = "ai-sage/GigaAM-v3"
HF_REVISIONS: dict[str, str] = {
    "v3_e2e_rnnt": "7655ad717f8122257385bb4b2f373db3697e8680",
    "v3_e2e_ctc": "cec030b4c4f35d928e4a9044a3bdb29ebd499fac",
    "v3_rnnt": "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8",
    "v3_ctc": "15ef3b5a88da78f93134b3cb7f015c70aefa8946",
}
_INSTALL_HINT = (
    "Run:\n"
    '  pip install "pipecat-ru[gigaam]"\n'
    '  pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"'
)


def load_gigaam_model(
    model_name: str = DEFAULT_MODEL,
    *,
    source: str = "auto",
    device: str | None = None,
    fp16_encoder: bool = True,
    use_flash: bool = False,
    download_root: str | None = None,
    hf_revision: str | None = None,
) -> Any:
    """Load a GigaAM model (downloads it on first use).

    Args:
        model_name: ``v3_e2e_rnnt``, ``v3_e2e_ctc``, ``v3_rnnt`` or ``v3_ctc``.
        source: ``"package"``: the ``gigaam`` package and SberDevices' CDN;
            ``"hf"``: the Hugging Face mirror (``ai-sage/GigaAM-v3``, pinned revision,
            weights only: no remote code);
            ``"auto"`` (default): the package, falling back to Hugging Face when the
            package is missing or its download fails.
        device: ``"cuda"``, ``"cpu"`` or None (GPU if available).
        fp16_encoder: Run the encoder in fp16 on the GPU (package source).
        use_flash: Use flash attention (package source; needs ``flash-attn``).
        download_root: Checkpoint cache of the ``gigaam`` package.
        hf_revision: Override the pinned Hugging Face revision.
    """
    if source not in ("auto", "package", "hf"):
        raise ValueError(f"source must be 'auto', 'package' or 'hf', not {source!r}")
    if source in ("auto", "package"):
        try:
            import gigaam
        except ModuleNotFoundError as e:
            if source == "package":
                raise ImportError(f"GigaAM is not installed. {_INSTALL_HINT}") from e
            logger.info("gigaam package not installed; loading GigaAM from Hugging Face")
        else:
            kwargs: dict[str, Any] = {"fp16_encoder": fp16_encoder, "use_flash": use_flash}
            if device is not None:
                kwargs["device"] = device
            if download_root is not None:
                kwargs["download_root"] = download_root
            try:
                return gigaam.load_model(model_name, **kwargs)
            except (RuntimeError, OSError) as e:
                # e.g. the CDN's TLS certificate expired (seen in November 2026)
                if source == "package":
                    raise
                logger.warning(f"GigaAM download failed ({e}); trying Hugging Face {HF_REPO}")
    return _load_from_hf(
        model_name,
        device=device,
        revision=hf_revision,
        fp16_encoder=fp16_encoder,
        use_flash=use_flash,
    )


def _load_from_hf(
    model_name: str,
    *,
    device: str | None,
    revision: str | None,
    fp16_encoder: bool = True,
    use_flash: bool = False,
) -> Any:
    """Build a ``gigaam`` model from the Hugging Face mirror's weights.

    Only data is downloaded (``config.json``, ``pytorch_model.bin``, ``tokenizer.model``
    at a pinned commit); the repo's modeling code is not run. The config's
    ``modeling_gigaam.*`` targets are mapped onto the ``gigaam`` package's classes.
    """
    if revision is None:
        if model_name not in HF_REVISIONS:
            raise ValueError(
                f"No pinned Hugging Face revision for {model_name!r}; pass hf_revision="
            )
        revision = HF_REVISIONS[model_name]
    try:
        import gigaam.decoder
        import gigaam.decoding
        import gigaam.encoder
        import gigaam.preprocess
        import torch
        from gigaam.model import GigaAMASR
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf
    except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
        raise ImportError(f"GigaAM is not installed. {_INSTALL_HINT}") from e

    def fetch(name: str) -> str:
        return str(hf_hub_download(HF_REPO, name, revision=revision))

    with open(fetch("config.json"), encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)["cfg"]["model"]["cfg"]
    modules = (gigaam.preprocess, gigaam.encoder, gigaam.decoder, gigaam.decoding)
    _retarget(cfg, {m.__name__: m for m in modules})
    if "decoding" in cfg and "model_path" in cfg["decoding"]:
        cfg["decoding"]["model_path"] = fetch("tokenizer.model")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.setdefault("encoder", {})["flash_attn"] = bool(use_flash) and target.type != "cpu"

    model = GigaAMASR(OmegaConf.create(cfg))
    state = torch.load(fetch("pytorch_model.bin"), map_location="cpu", weights_only=True)
    prefix = "model."
    state = {k.removeprefix(prefix): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.eval()
    if fp16_encoder and target.type != "cpu":
        model.encoder = model.encoder.half()
    return model.to(target)


def _retarget(node: Any, modules: dict[str, Any]) -> None:
    """Point ``_target_: modeling_gigaam.X`` at the ``gigaam`` module defining X."""
    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str) and target.startswith("modeling_gigaam."):
            cls = target.split(".", 1)[1]
            owner = next((name for name, m in modules.items() if hasattr(m, cls)), None)
            if owner is None:
                raise ValueError(f"gigaam package has no class {cls!r} for {target!r}")
            node["_target_"] = f"{owner}.{cls}"
        for value in node.values():
            _retarget(value, modules)
    elif isinstance(node, list):
        for value in node:
            _retarget(value, modules)


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
        # The gigaam package decodes with model._decode(); the Hugging Face modeling
        # code with model.decoding.decode(model.head, ...). Both take tensors.
        self.fast_path = (
            hasattr(model, "forward")
            and hasattr(model, "_device")
            and (
                hasattr(model, "_decode") or (hasattr(model, "decoding") and hasattr(model, "head"))
            )
        )

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
            if hasattr(model, "_decode"):
                decoded = model._decode(encoded, encoded_len, length, False)[0]
            else:
                decoded = model.decoding.decode(model.head, encoded, encoded_len)[0]
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
        source: Where to load the checkpoint from (see :func:`load_gigaam_model`).
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
        source: str = "auto",
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
        self._source = source
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
                source=self._source,
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
