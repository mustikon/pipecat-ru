from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipecat_ru import bench
from pipecat_ru._audio import float_to_pcm16, pcm16_to_float, split_long
from tests.conftest import FakeGigaAM, FakeTensor


def write_wav(path: Path, secs: float, sr: int = 16000) -> None:
    sf.write(str(path), np.zeros(int(secs * sr), dtype=np.float32), sr)


@pytest.fixture
def phrases_dir(tmp_path: Path) -> Path:
    write_wav(tmp_path / "001.wav", 1.0)
    (tmp_path / "001.txt").write_text("Привет, как дела?", encoding="utf-8")
    write_wav(tmp_path / "002.wav", 0.5, sr=48000)
    write_wav(tmp_path / "003.flac", 0.5)
    (tmp_path / "transcripts.tsv").write_text(
        "# file\ttext\n002.wav\tВключи свет на кухне.\n003\tМне 21 год.\n", encoding="utf-8"
    )
    write_wav(tmp_path / "004.wav", 0.5)  # no reference: skipped
    return tmp_path


def test_normalize_for_wer() -> None:
    assert bench.normalize_for_wer("Ёлка, ёж — и всё!") == "елка еж и все"
    assert bench.normalize_for_wer("Мне 21 год.") == "мне двадцать один год"
    assert bench.normalize_for_wer("Мне 21 год.", spell_numbers=False) == "мне 21 год"
    assert bench.normalize_for_wer("северо-западный") == "северо-западный"


def test_edit_distance() -> None:
    assert bench.edit_distance("kitten", "sitting") == 3
    assert bench.edit_distance([], ["a"]) == 1
    assert bench.edit_distance(["a", "b"], ["a", "b"]) == 0


def test_find_phrases(phrases_dir: Path) -> None:
    found = bench.find_phrases(phrases_dir)
    assert [(p.path.name, p.reference) for p in found] == [
        ("001.wav", "Привет, как дела?"),
        ("002.wav", "Включи свет на кухне."),
        ("003.flac", "Мне 21 год."),
    ]


def test_load_audio_resamples_to_16k(phrases_dir: Path) -> None:
    audio = bench.load_audio(phrases_dir / "002.wav")
    assert audio.dtype == np.float32
    assert len(audio) == 8000


class ScriptedTranscriber:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0

    def transcribe(self, samples: np.ndarray) -> str:
        answer = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return answer


def test_run_stt_benchmark(phrases_dir: Path) -> None:
    phrases = bench.find_phrases(phrases_dir)
    tr = ScriptedTranscriber(["привет как дела", "включи свет в кухне", "Мне двадцать один год"])
    summary = bench.run_stt_benchmark(tr, phrases, warmup=0)
    assert summary.phrases == 3
    assert [r.wer for r in summary.rows] == [0.0, 0.25, 0.0]
    assert summary.wer == round(1 / 11, 4)  # 3 + 4 + 4 reference words
    assert summary.latency_max_ms >= summary.latency_p50_ms >= 0
    assert tr.calls == 3


def test_run_stt_benchmark_warmup_is_untimed(phrases_dir: Path) -> None:
    tr = ScriptedTranscriber(["x"])
    bench.run_stt_benchmark(tr, bench.find_phrases(phrases_dir), warmup=2)
    assert tr.calls == 5


def test_run_tts_benchmark(tmp_path: Path) -> None:
    seen: list[str] = []

    def synth(text: str) -> tuple[FakeTensor, int]:
        seen.append(text)
        return FakeTensor(np.zeros(2400, dtype=np.float32)), 24000

    summary = bench.run_tts_benchmark(
        synth, ["Хорошо, сейчас посмотрю. Готово."], warmup=0, save_dir=tmp_path
    )
    assert seen == ["Хорошо, сейчас посмотрю.", "Готово."]
    row = summary.rows[0]
    assert row.chunks == 2 and row.audio_s == 0.2
    assert row.first_audio_ms <= row.total_ms
    assert (tmp_path / "001.wav").exists()


def test_cli_stt(
    phrases_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pipecat_ru.stt.gigaam as gigaam_mod

    monkeypatch.setattr(gigaam_mod, "load_gigaam_model", lambda *a, **k: FakeGigaAM("привет"))
    out_csv = tmp_path / "out.csv"
    assert bench.main(["stt", str(phrases_dir), "--json", "--csv", str(out_csv)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["phrases"] == 3
    assert len(data["rows"]) == 3
    assert out_csv.read_text(encoding="utf-8").startswith("file,duration_s,latency_ms")


def test_cli_stt_empty_folder(tmp_path: Path) -> None:
    assert bench.main(["stt", str(tmp_path)]) == 2


def test_cli_tts_silero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pipecat_ru.tts.silero as silero_mod
    from tests.conftest import FakeSileroModel

    monkeypatch.setattr(silero_mod, "load_silero_tts", lambda *a, **k: FakeSileroModel())
    assert bench.main(["tts", "--engine", "silero", "--warmup", "0"]) == 0
    out = capsys.readouterr().out
    assert "first_audio_p50_ms" in out


def test_pcm_roundtrip() -> None:
    samples = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    back = pcm16_to_float(float_to_pcm16(samples))
    assert np.allclose(back, [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-3)
    assert float_to_pcm16(FakeTensor(np.zeros(3, np.float32))) == b"\x00\x00" * 3


def test_split_long_cuts_at_quiet_points() -> None:
    sr = 1000
    loud = np.ones(sr * 10, dtype=np.float32)
    loud[7000:7100] = 0.0  # a pause at 7 s
    pieces = split_long(loud, sr, max_secs=8.0)
    assert len(pieces) == 2
    assert 7000 <= len(pieces[0]) <= 7100
    assert sum(len(p) for p in pieces) == len(loud)
    assert split_long(loud[:100], sr, 8.0)[0] is not None
