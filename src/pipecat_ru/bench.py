"""Latency and accuracy benchmark on your own recordings.

STT (GigaAM): a folder of recorded phrases, each with its reference text::

    phrases/
      001.wav   001.txt      # sidecar .txt with what was said
      002.wav   002.txt
      ...                    # or one transcripts.tsv: "<file name>\\t<text>" per line

    pipecat-ru-bench stt phrases/ --device cuda --csv stt.csv

reports WER, CER and per-phrase latency (p50/p90/max) and real-time factor.

TTS (Qwen3-TTS or Silero): time to first audio (normalize + split + synthesize the
first chunk), total synthesis time and real-time factor::

    pipecat-ru-bench tts --engine qwen3 --ref-audio me.wav --ref-text "..." --save-dir out/
    pipecat-ru-bench tts --engine silero --texts phrases.txt

Texts come from ``--texts`` (one per line) or a built-in Russian set.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipecat_ru._audio import FloatArray
from pipecat_ru.text.normalizer import RussianTextNormalizer
from pipecat_ru.text.splitter import RussianTextSplitter

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")
STT_SAMPLE_RATE = 16000

DEFAULT_TTS_TEXTS = [
    "Привет! Чем могу помочь?",
    "Хорошо, сейчас посмотрю, что там с погодой на завтра.",
    "Встреча перенесена на 15 марта, в 14:30, т.е. через 2 дня.",
    "Я запустил Claude Code в фоне; как только он закончит, я сообщу результат.",
    "Температура в Москве минус 5 градусов, ветер северо-западный, 7 метров в секунду.",
    "Готово. Я нашёл три подходящих варианта, давай расскажу про каждый по порядку.",
]

# --------------------------------------------------------------------- metrics

_PUNCT_RE = re.compile(r"[^\w\s-]|(?<!\w)-|-(?!\w)", re.UNICODE)


def normalize_for_wer(text: str, *, spell_numbers: bool = True) -> str:
    """Lowercase, fold "ё", spell out numbers, drop punctuation: the WER view of text."""
    if spell_numbers and any(c.isdigit() for c in text):
        text = RussianTextNormalizer(use_default_lexicon=False, restore_yo=False)(text)
    text = text.lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences (words or characters)."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation (q in 0..100); 0.0 for no values."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# --------------------------------------------------------------------- data


@dataclass
class Phrase:
    """One recorded phrase with its reference text."""

    path: Path
    reference: str


def find_phrases(folder: Path) -> list[Phrase]:
    """Collect audio files with references from sidecar .txt or transcripts.tsv."""
    table: dict[str, str] = {}
    tsv = folder / "transcripts.tsv"
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if "\t" in line and not line.startswith("#"):
                name, text = line.split("\t", 1)
                table[name.strip()] = text.strip()
    phrases: list[Phrase] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            ref = sidecar.read_text(encoding="utf-8").strip()
        else:
            ref = table.get(path.name, table.get(path.stem, ""))
        if ref:
            phrases.append(Phrase(path, ref))
        else:
            print(f"skip {path.name}: no reference text", file=sys.stderr)
    return phrases


def load_audio(path: Path, sample_rate: int = STT_SAMPLE_RATE) -> FloatArray:
    """Read an audio file as mono float32 at ``sample_rate`` (linear resampling)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return resample_linear(mono, int(sr), sample_rate)


def resample_linear(samples: FloatArray, src_rate: int, dst_rate: int) -> FloatArray:
    """Plain linear-interpolation resampling: adequate for ASR benchmarking."""
    if src_rate == dst_rate or len(samples) == 0:
        return samples.astype(np.float32)
    n_out = round(len(samples) * dst_rate / src_rate)
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# --------------------------------------------------------------------- STT


class Transcriber(Protocol):
    """Anything that turns 16 kHz float32 audio into text."""

    def transcribe(self, samples: FloatArray) -> str:
        """Return the transcript."""
        ...


@dataclass
class SttRow:
    """Per-phrase STT result."""

    file: str
    duration_s: float
    latency_ms: float
    rtf: float
    wer: float
    cer: float
    reference: str
    hypothesis: str


@dataclass
class SttSummary:
    """Corpus-level STT result."""

    phrases: int
    wer: float
    cer: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_max_ms: float
    mean_rtf: float
    rows: list[SttRow] = field(default_factory=list)


def run_stt_benchmark(
    transcriber: Transcriber,
    phrases: Iterable[Phrase],
    *,
    warmup: int = 1,
    loader: Callable[[Path], FloatArray] = load_audio,
    spell_numbers: bool = True,
) -> SttSummary:
    """Transcribe every phrase and compute WER/CER and latency."""
    items = [(p, loader(p.path)) for p in phrases]
    for _, audio in items[:warmup]:
        transcriber.transcribe(audio)
    rows: list[SttRow] = []
    word_err = word_total = char_err = char_total = 0
    for phrase, audio in items:
        t0 = time.perf_counter()
        hyp = transcriber.transcribe(audio)
        latency = time.perf_counter() - t0
        duration = len(audio) / STT_SAMPLE_RATE
        ref_n = normalize_for_wer(phrase.reference, spell_numbers=spell_numbers)
        hyp_n = normalize_for_wer(hyp, spell_numbers=spell_numbers)
        w_err, w_tot = edit_distance(ref_n.split(), hyp_n.split()), len(ref_n.split())
        c_ref, c_hyp = ref_n.replace(" ", ""), hyp_n.replace(" ", "")
        c_err, c_tot = edit_distance(c_ref, c_hyp), len(c_ref)
        word_err, word_total = word_err + w_err, word_total + w_tot
        char_err, char_total = char_err + c_err, char_total + c_tot
        rows.append(
            SttRow(
                file=phrase.path.name,
                duration_s=round(duration, 3),
                latency_ms=round(latency * 1000, 1),
                rtf=round(latency / duration, 4) if duration else 0.0,
                wer=round(w_err / w_tot, 4) if w_tot else 0.0,
                cer=round(c_err / c_tot, 4) if c_tot else 0.0,
                reference=phrase.reference,
                hypothesis=hyp,
            )
        )
    latencies = [r.latency_ms for r in rows]
    return SttSummary(
        phrases=len(rows),
        wer=round(word_err / word_total, 4) if word_total else 0.0,
        cer=round(char_err / char_total, 4) if char_total else 0.0,
        latency_p50_ms=round(percentile(latencies, 50), 1),
        latency_p90_ms=round(percentile(latencies, 90), 1),
        latency_max_ms=round(max(latencies, default=0.0), 1),
        mean_rtf=round(statistics.fmean([r.rtf for r in rows]), 4) if rows else 0.0,
        rows=rows,
    )


# --------------------------------------------------------------------- TTS

Synthesizer = Callable[[str], tuple[Any, int]]


@dataclass
class TtsRow:
    """Per-text TTS result."""

    text: str
    chunks: int
    first_chunk: str
    first_audio_ms: float
    total_ms: float
    audio_s: float
    rtf: float


@dataclass
class TtsSummary:
    """TTS result over all texts."""

    texts: int
    first_audio_p50_ms: float
    first_audio_p90_ms: float
    first_audio_max_ms: float
    mean_rtf: float
    rows: list[TtsRow] = field(default_factory=list)


def _to_float_array(samples: Any) -> FloatArray:
    if hasattr(samples, "detach"):
        samples = samples.detach().float().cpu().numpy()
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def run_tts_benchmark(
    synthesize: Synthesizer,
    texts: Iterable[str],
    *,
    normalizer: RussianTextNormalizer | None = None,
    warmup: int = 1,
    save_dir: Path | None = None,
) -> TtsSummary:
    """Synthesize each text chunk by chunk, as the Pipecat service would."""
    normalizer = normalizer if normalizer is not None else RussianTextNormalizer()
    texts = list(texts)
    for text in texts[:warmup]:
        synthesize(normalizer(text))
    rows: list[TtsRow] = []
    for idx, text in enumerate(texts):
        t0 = time.perf_counter()
        chunks = RussianTextSplitter().split(text)
        first_ms = 0.0
        audio_parts: list[FloatArray] = []
        sample_rate = 0
        for n, chunk in enumerate(chunks):
            samples, sample_rate = synthesize(normalizer(chunk))
            audio_parts.append(_to_float_array(samples))
            if n == 0:
                first_ms = (time.perf_counter() - t0) * 1000
        total = time.perf_counter() - t0
        audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, np.float32)
        audio_s = len(audio) / sample_rate if sample_rate else 0.0
        if save_dir is not None and sample_rate:
            import soundfile as sf

            save_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_dir / f"{idx + 1:03d}.wav"), audio, sample_rate)
        rows.append(
            TtsRow(
                text=text,
                chunks=len(chunks),
                first_chunk=chunks[0] if chunks else "",
                first_audio_ms=round(first_ms, 1),
                total_ms=round(total * 1000, 1),
                audio_s=round(audio_s, 3),
                rtf=round(total / audio_s, 4) if audio_s else 0.0,
            )
        )
    firsts = [r.first_audio_ms for r in rows]
    return TtsSummary(
        texts=len(rows),
        first_audio_p50_ms=round(percentile(firsts, 50), 1),
        first_audio_p90_ms=round(percentile(firsts, 90), 1),
        first_audio_max_ms=round(max(firsts, default=0.0), 1),
        mean_rtf=round(statistics.fmean([r.rtf for r in rows]), 4) if rows else 0.0,
        rows=rows,
    )


# --------------------------------------------------------------------- CLI


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _print_summary(summary: SttSummary | TtsSummary, as_json: bool) -> None:
    data = asdict(summary)
    rows = data.pop("rows")
    if as_json:
        print(json.dumps({**data, "rows": rows}, ensure_ascii=False, indent=2))
        return
    for row in rows:
        if isinstance(summary, SttSummary):
            mark = "" if row["wer"] == 0 else f"  <- {row['hypothesis']!r}"
            print(f"{row['file']:<28} {row['latency_ms']:>8.1f} ms  WER {row['wer']:.2f}{mark}")
        else:
            print(
                f"first {row['first_audio_ms']:>7.1f} ms  total {row['total_ms']:>7.1f} ms  "
                f"RTF {row['rtf']:.3f}  [{row['first_chunk']}]"
            )
    print("-" * 60)
    for key, value in data.items():
        print(f"{key:<22} {value}")


def _cmd_stt(args: argparse.Namespace) -> int:
    from pipecat_ru.stt.gigaam import GigaAMTranscriber, load_gigaam_model

    phrases = find_phrases(Path(args.folder))
    if not phrases:
        print(f"no phrases with reference text in {args.folder}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    model = load_gigaam_model(args.model, device=args.device)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    summary = run_stt_benchmark(
        GigaAMTranscriber(model), phrases, warmup=args.warmup, spell_numbers=not args.raw
    )
    if args.csv:
        _write_csv(Path(args.csv), summary.rows)
    _print_summary(summary, args.json)
    return 0


def _cmd_tts(args: argparse.Namespace) -> int:
    texts = DEFAULT_TTS_TEXTS
    if args.texts:
        texts = [t for t in Path(args.texts).read_text(encoding="utf-8").splitlines() if t.strip()]
    service: Any
    if args.engine == "qwen3":
        from pipecat_ru.tts.qwen3 import Qwen3TTSService

        service = Qwen3TTSService(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only=args.ref_audio is not None and args.ref_text is None,
            speaker=args.speaker,
            model_name=args.model,
            device=args.device or "cuda:0",
            attn_implementation=args.attn,
        )
    else:
        from pipecat_ru.tts.silero import SileroTTSService

        service = SileroTTSService(
            speaker=args.speaker or "xenia",
            model_id=args.model or "v5_ru",
            device=args.device or "cpu",
        )
    t0 = time.perf_counter()
    service.load_model()
    print(f"loaded {args.engine} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    summary = run_tts_benchmark(
        service.synthesize,
        texts,
        normalizer=service.normalizer or RussianTextNormalizer(),
        warmup=args.warmup,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )
    if args.csv:
        _write_csv(Path(args.csv), summary.rows)
    _print_summary(summary, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of ``pipecat-ru-bench``."""
    parser = argparse.ArgumentParser(prog="pipecat-ru-bench", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    stt = sub.add_parser("stt", help="GigaAM WER and latency on a folder of recordings")
    stt.add_argument("folder", help="folder with audio files and reference texts")
    stt.add_argument("--model", default="v3_e2e_rnnt", help="GigaAM model name")
    stt.add_argument("--device", default=None, help="cuda / cpu (default: GigaAM's choice)")
    stt.add_argument("--raw", action="store_true", help="do not spell out numbers before WER")
    stt.set_defaults(func=_cmd_stt)

    tts = sub.add_parser("tts", help="time to first audio and RTF of a TTS engine")
    tts.add_argument("--engine", choices=["qwen3", "silero"], default="qwen3")
    tts.add_argument("--texts", help="text file, one phrase per line")
    tts.add_argument("--model", help="model id (Qwen HF id or Silero model id)")
    tts.add_argument("--device", help="cuda:0 / cpu")
    tts.add_argument("--ref-audio", help="Qwen3: reference clip for voice cloning")
    tts.add_argument("--ref-text", help="Qwen3: transcript of the reference clip")
    tts.add_argument("--speaker", help="Qwen3 CustomVoice speaker or Silero speaker")
    tts.add_argument("--attn", help="Qwen3: attention implementation, e.g. flash_attention_2")
    tts.add_argument("--save-dir", help="write the synthesized audio here")
    tts.set_defaults(func=_cmd_tts)

    for p in (stt, tts):
        p.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs")
        p.add_argument("--csv", help="write per-item results to this CSV file")
        p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``pipecat-ru-bench``."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
