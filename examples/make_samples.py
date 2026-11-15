"""Synthesize the same Russian phrases with every engine, to compare by ear.

    python examples/make_samples.py --out samples
    python examples/make_samples.py --ref-audio me.wav --ref-text "Текст записи."

Writes ``<engine>_<n>.wav`` files and ``timings.json`` (time to first audio of the
first chunk, total synthesis time, audio length). Without ``--ref-audio`` the Qwen3
voice clone uses a Silero voice as its reference, which is only good for a demo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pipecat_ru.text import RussianTextNormalizer, RussianTextSplitter, StressFormat, StressMarker

PHRASES = [
    # numbers, dates, money, abbreviations
    "Встреча перенесена на 15.03.2026 в 14:30, т.е. через 2 дня; бюджет — 1 250 000 руб.",
    # homographs: замок / замок, мука / мука, атлас / атлас
    "Мы подошли к старому замку и увидели, что замок на воротах сломан. "
    "Мука с сахаром закончилась, и это была настоящая мука.",
    # what an assistant says
    "Хорошо, я запустил Claude Code в фоне. Как только он закончит, я сообщу результат.",
]
REF_TEXT = "Привет! Это пример моего голоса, по нему модель научится говорить так же."


def to_np(samples: Any) -> np.ndarray:
    if hasattr(samples, "detach"):
        samples = samples.detach().float().cpu().numpy()
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def synth_reply(
    name: str,
    synthesize: Any,
    normalizer: RussianTextNormalizer,
    text: str,
    out: Path,
) -> dict[str, Any]:
    """Synthesize like the Pipecat service: split, normalize, one call per chunk."""
    chunks = RussianTextSplitter().split(text)
    parts: list[np.ndarray] = []
    sr = 0
    t0 = time.perf_counter()
    first_ms = 0.0
    spoken: list[str] = []
    for i, chunk in enumerate(chunks):
        norm = normalizer(chunk)
        spoken.append(norm)
        samples, sr = synthesize(norm)
        parts.append(to_np(samples))
        if i == 0:
            first_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    audio = np.concatenate(parts)
    sf.write(str(out / f"{name}.wav"), audio, sr)
    result = {
        "file": f"{name}.wav",
        "first_audio_ms": round(first_ms),
        "total_ms": round(total_ms),
        "audio_s": round(len(audio) / sr, 2),
        "chunks": chunks,
        "spoken_text": spoken,
    }
    audio_s = len(audio) / sr
    print(f"{name:<28} first {first_ms:7.0f} ms  total {total_ms:7.0f} ms  audio {audio_s:5.1f} s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="samples")
    parser.add_argument("--ref-audio", help="your voice for the Qwen3 clone (5-15 s WAV)")
    parser.add_argument("--ref-text", help="exact transcript of --ref-audio")
    parser.add_argument("--speaker", default="Ryan", help="Qwen3 CustomVoice speaker")
    parser.add_argument("--skip-clone", action="store_true", help="skip the Qwen3 Base model")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from pipecat_ru.tts.qwen3 import Qwen3TTSService
    from pipecat_ru.tts.silero import SileroTTSService

    plain = RussianTextNormalizer()
    stress = StressMarker(StressFormat.PLUS)
    stressed = RussianTextNormalizer(stress=stress)
    acute = RussianTextNormalizer(stress=StressMarker(StressFormat.ACUTE, accentizer=stress.load()))
    results: list[dict[str, Any]] = []

    def warm(synth: Any) -> None:
        synth("Проверка связи.")

    silero = SileroTTSService(speaker="xenia", silero_sample_rate=48000)
    silero.load_model()
    warm(silero.synthesize)
    for i, text in enumerate(PHRASES, 1):
        results.append(synth_reply(f"silero_xenia_{i}", silero.synthesize, plain, text, out))
    results.append(
        synth_reply("silero_xenia_2_ruaccent", silero.synthesize, stressed, PHRASES[1], out)
    )

    ref_audio, ref_text = args.ref_audio, args.ref_text
    if not ref_audio:
        ref = SileroTTSService(speaker="baya", silero_sample_rate=24000)
        ref.load_model()
        samples, sr = ref.synthesize(REF_TEXT)
        ref_audio = str(out / "reference_silero_baya.wav")
        sf.write(ref_audio, to_np(samples), sr)
        ref_text = REF_TEXT

    qwen = Qwen3TTSService(speaker=args.speaker)
    qwen.load_model()
    warm(qwen.synthesize)
    for i, text in enumerate(PHRASES, 1):
        results.append(
            synth_reply(f"qwen3_{args.speaker.lower()}_{i}", qwen.synthesize, plain, text, out)
        )
    results.append(
        synth_reply(
            f"qwen3_{args.speaker.lower()}_2_acute", qwen.synthesize, acute, PHRASES[1], out
        )
    )
    del qwen
    _free_gpu()

    if not args.skip_clone:
        clone = Qwen3TTSService(ref_audio=ref_audio, ref_text=ref_text)
        clone.load_model()
        warm(clone.synthesize)
        for i, text in enumerate(PHRASES, 1):
            results.append(synth_reply(f"qwen3_clone_{i}", clone.synthesize, plain, text, out))

    (out / "timings.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: S110 - best effort
        pass


if __name__ == "__main__":
    main()
