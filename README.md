# pipecat-ru

[![CI](https://github.com/mustikon/pipecat-ru/actions/workflows/ci.yml/badge.svg)](https://github.com/mustikon/pipecat-ru/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pipecat-ru.svg)](https://pypi.org/project/pipecat-ru/)
[![Python](https://img.shields.io/pypi/pyversions/pipecat-ru.svg)](https://pypi.org/project/pipecat-ru/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Russian-language plugins for the [Pipecat](https://github.com/pipecat-ai/pipecat) voice
framework: local STT and TTS, plus the text and turn-taking pieces that make a Russian voice
assistant feel natural.**

[Русская версия](README.ru.md)

Pipecat's built-in services and defaults are tuned for English. Out of the box, a Russian bot
reads "15.03 в 14:30" digit by digit and waits for a whole sentence before it speaks. It also
stops mid-sentence whenever you say "угу". This package fixes that with open models that run
on your own GPU.

| Piece | What it does |
|---|---|
| `GigaAMSTTService` | Final Russian transcripts with punctuation from [GigaAM v3](https://github.com/salute-developers/GigaAM). Stronger on Russian than Whisper, and it does not hallucinate on silence. |
| `Qwen3TTSService` | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) speech, with voice cloning from a short Russian reference clip. |
| `SileroTTSService` | [Silero v5](https://github.com/snakers4/silero-models): fast CPU fallback voice. **Weights are non-commercial (CC BY-NC 4.0).** |
| `RussianTextNormalizer` | Spells out numbers, dates, times, money, units and abbreviations (`т.е.`, `ул.`, `в 2024 г.`). Applies a pronunciation lexicon and optional stress marks via [ruaccent](https://github.com/Den4ikAI/ruaccent). |
| `RussianTextAggregator` | Sentence/clause splitter that makes the **first** chunk of every reply a short clause, so the first audio starts sooner. |
| `BackchannelUserTurnStartStrategy` | While the bot speaks, short "ага / угу / да / так" (under ~600 ms) do not interrupt it, but "стоп" still does. Pairs with smart-turn v3.2, which understands Russian. |
| `pipecat-ru-bench` | WER and latency on a folder of your own recordings. Time to first audio for the TTS engines. |

## Install

```bash
pip install pipecat-ru                       # text tools + turn strategy (no torch)
pip install "pipecat-ru[gigaam,qwen,silero,accent]"
pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
```

GigaAM v3 is not on PyPI (the `gigaam` name there is an old v1 build), so it is installed
from GitHub, with `--no-deps`: GigaAM pins `onnxruntime==1.23.*`, which conflicts with
pipecat-ai's `onnxruntime>=1.24.3`, and it only needs onnxruntime for ONNX export. The
`gigaam` extra installs the rest of its dependencies. GigaAM also needs `ffmpeg` on `PATH` for its file-based API. This package feeds
it tensors directly, so `ffmpeg` is only used as a fallback.

**RTX 50xx (Blackwell) on Windows:** install a CUDA 12.8+ build of PyTorch *before* the extras,
for example `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.
Older wheels do not support `sm_120`. `flash-attn` is optional. Pass
`attn_implementation="flash_attention_2"` to `Qwen3TTSService` only if it is installed.

> `qwen-tts` pins `transformers==4.57.3`. Do not install `pipecat-ai[local-smart-turn]` in the
> same environment, because it requires transformers 5.x. You do not need it: smart-turn v3.2
> runs on pipecat-ai's core ONNX runtime.

## Quick start

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat_ru.stt.gigaam import GigaAMSTTService
from pipecat_ru.tts.qwen3 import Qwen3TTSService
from pipecat_ru.turns import russian_user_turn_strategies

stt = GigaAMSTTService(device="cuda")
tts = Qwen3TTSService(ref_audio="my_voice.wav", ref_text="Точный текст этой записи.")
aggregators = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        user_turn_strategies=russian_user_turn_strategies(),
    ),
)
pipeline = Pipeline([transport.input(), stt, aggregators.user(), llm, tts,
                     transport.output(), aggregators.assistant()])
```

[`examples/local_bot.py`](examples/local_bot.py) is a complete microphone-to-speaker bot. It
uses a local OpenAI-compatible LLM and Silero as automatic failover for Qwen3-TTS.

## Components

### GigaAM STT

`GigaAMSTTService` is a Pipecat `SegmentedSTTService`: the VAD cuts speech into utterances,
and each one is transcribed in a worker thread. The default model is `v3_e2e_rnnt`, which gives
punctuation and normalized text. `v3_e2e_ctc` is faster. Utterances over 24 s are split at
quiet points, because GigaAM's short-form API stops at 25 s. You can pass a loaded `model=` to
share one model between services. Otherwise the model loads when the pipeline starts.

Checkpoints normally come from SberDevices' CDN through the `gigaam` package. If that download
fails (in November 2026 the CDN served an expired TLS certificate), the loader falls back to the
official Hugging Face mirror `ai-sage/GigaAM-v3`. It fetches only the weights, the config and the
tokenizer at a pinned commit; no remote code runs. Choose the source explicitly with
`GigaAMSTTService(source="package" | "hf" | "auto")`.

### TTS: Qwen3-TTS and Silero

Both services synthesize one text chunk per call in a worker thread, and stream the audio out
in 200 ms frames.

* **Qwen3-TTS voice clone:** `Qwen3TTSService(ref_audio=..., ref_text=...)` with the 1.7B or
  0.6B *Base* model. The clone prompt is built once at startup.
* **Qwen3-TTS preset speaker:** `Qwen3TTSService(speaker="Ryan", instruct="...")` with a
  *CustomVoice* model.
* **Silero:** `SileroTTSService(speaker="xenia")`, with speakers `aidar`, `baya`, `kseniya`,
  `xenia` and `eugene`. It runs on the CPU. Silero reads `+` stress marks natively.

The `qwen-tts` package does not expose incremental audio yet. Time to first audio is therefore
set by the length of the first chunk, and keeping that chunk short is the splitter's job.

Both services install the Russian splitter and normalizer by default:
`normalizer=None` / `splitter=None` turn them off, or pass your own.

### Text normalizer

```python
from pipecat_ru.text import RussianTextNormalizer, StressMarker

n = RussianTextNormalizer(lexicon={"Мустикон": "Мустик+он"})
n("Встреча 15.03.2025 в 14:30, т.е. через 2 дня.")
# 'Встреча пятнадцатое марта две тысячи двадцать пятого года в четырнадцать часов
#  тридцать минут, то есть через два дня.'
n("Работаем с 9:00 до 18:00, ул. Ленина д. 7")
# 'Работаем с девяти до восемнадцати, улица Ленина дом семь'

with_stress = RussianTextNormalizer(stress=StressMarker("plus"))   # з+амок (Silero)
```

The steps run in order:

1. A lexicon of names the TTS would mangle (`Claude Code` → `Клод Код`, `GitHub` → `Гитхаб`).
   It is extensible and can be switched off.
2. Russian pre-rules for cases rutextnorm misses: abbreviations, `в 2024 г.`, time ranges in
   the genitive, ages, `версия 3.12`.
3. [rutextnorm](https://pypi.org/project/rutextnorm/) for numbers, dates, money and units, with
   case agreement.
4. Optional ruaccent stress marks, written as `+` (Silero), as a combining acute accent, or
   stripped.

To use the normalizer with any Pipecat TTS, pass
`text_transforms=[("*", normalizer.transform)]`.

### Fast first chunk

`RussianTextSplitter` / `RussianTextAggregator` release the first chunk of a reply at the
first `,` `;` `:` or dash, once it holds 3 words. After that they release whole sentences, and
cut a sentence longer than 220 characters at a clause boundary. Abbreviations, initials and
decimals never split a chunk. Tune it with `SplitterConfig`.

```python
RussianTextSplitter().split("Хорошо, сейчас посмотрю, что там с погодой. Завтра ясно.")
# ['Хорошо, сейчас посмотрю,', 'что там с погодой.', 'Завтра ясно.']
```

### Backchannels do not interrupt

`russian_user_turn_strategies()` returns Pipecat `UserTurnStrategies`. The start strategy is
`BackchannelUserTurnStartStrategy` and the stop strategy is smart-turn v3.2
(`LocalSmartTurnAnalyzerV3`).

The start strategy replaces Pipecat's default VAD and transcription start strategies:

* **Bot silent:** a turn starts on speech, as before.
* **Bot speaking:** an utterance interrupts only if it lasts longer than `max_backchannel_secs`
  (0.6 s) or its transcript has a word that is not a backchannel. A short "ага" is dropped from
  the context, and the bot keeps talking.

Set `vad_stop_secs` to the `stop_secs` of your VAD. Edit the vocabulary with `backchannels=`.

### Benchmark on your own voice

```bash
# folder: 001.wav + 001.txt (reference), ... or a transcripts.tsv "file<TAB>text"
pipecat-ru-bench stt my_phrases/ --device cuda --csv stt.csv
pipecat-ru-bench tts --engine qwen3 --ref-audio me.wav --ref-text "..." --save-dir out/
pipecat-ru-bench tts --engine silero
```

STT reports WER and CER (numbers are spelled out on both sides before scoring), latency p50,
p90 and max, and the real-time factor. TTS reports time to first audio and the RTF.

## Measured performance

Measured on an RTX 5090 (32 GB) with Windows 11, torch 2.11+cu128 and bf16. Numbers come from
`tests/test_gpu_integration.py` and `examples/make_samples.py`.

| Step | Result |
|---|---|
| GigaAM v3 e2e-RNNT, 4 s phrase | **66 ms**, transcript exact |
| Normalizer (no stress) | 0.25 ms per sentence |
| ruaccent stress (`turbo3.1`, CPU) | 12–60 ms per chunk |
| Silero v5 (CPU), first chunk | **60–140 ms**; a 10 s reply in 0.25 s |
| Qwen3-TTS 1.7B (`qwen-tts` 0.1.1, sdpa) | **RTF ≈ 1.4**: 3 s for 2 s of audio; first audio 4–10 s |

Qwen3-TTS through the official `qwen-tts` package is **slower than real time** on this GPU.
During generation the GPU is only about 8% busy: the bottleneck is per-step Python overhead in
the generation loop. The 0.6B model is no faster. For live replies, use Silero (or another fast
voice) for now. Qwen3-TTS becomes viable with a faster runtime, such as vLLM-Omni or a CUDA-graph
build of the decoder; that is on the roadmap.

ruaccent resolves common homographs («повесил зам+ок», «насыпь мук+у»). It can still pick the
wrong reading in ambiguous sentences («замок на воротах сломан»). Stress is added per TTS chunk,
so the model sees less context than in the whole reply.

## Model licences

The code in this repository is Apache-2.0. **The models have their own licences**, and you
accept them when you download the weights.

| Model / data | Used by | Licence | Commercial use |
|---|---|---|---|
| GigaAM v3 (SberDevices) | `GigaAMSTTService` | MIT | yes |
| Qwen3-TTS (Alibaba Qwen) | `Qwen3TTSService` | Apache-2.0 | yes |
| Silero TTS v5 | `SileroTTSService` | **CC BY-NC 4.0** | **no** (licence from the authors needed) |
| ruaccent | `StressMarker` (optional) | MIT (per its README) | yes |
| rutextnorm | `RussianTextNormalizer` | MIT | yes |
| smart-turn v3.2 (Pipecat) | `russian_user_turn_strategies` | BSD-2-Clause | yes |
| Silero VAD | Pipecat's `SileroVADAnalyzer` | MIT | yes |

Licences change. Check each model card before you ship anything.

## Tests

```bash
uv sync
uv run pytest                    # all models are mocked: no GPU, no downloads
```

Optional real-model tests for a CUDA machine (written for an RTX 5090 on Windows 11):

```powershell
uv sync --all-extras
uv pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
$env:PIPECAT_RU_GPU_TESTS = "1"
$env:PIPECAT_RU_REF_AUDIO = "C:\voice\me.wav"         # optional, for voice cloning
$env:PIPECAT_RU_REF_TEXT  = "Текст этой записи."
uv run pytest -m gpu -s
```

## Status and roadmap

Alpha, built for a personal assistant and published in case it helps others. Planned:

* a streaming partials STT (T-one or NVIDIA Nemotron streaming ASR), so backchannels and
  interruptions can be judged before the utterance ends;
* a real-time Qwen3-TTS runtime (vLLM-Omni or CUDA graphs) with incremental audio.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
