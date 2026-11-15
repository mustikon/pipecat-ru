# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Fast streaming generation for `Qwen3TTSService` (12 Hz models on CUDA, on by default;
  `fast=False` turns it off). The talker and code predictor run on static KV caches, a
  whole codec frame (15 code-predictor steps, one talker step and sampling) is replayed
  as one CUDA graph, and audio is decoded and streamed while later frames are generated.
  On an RTX 5090, the 1.7B model reaches first audio in about 85 ms with RTF about 0.25,
  against RTF 1.4 and 3-10 s through plain `qwen-tts`. Needs only torch and `qwen-tts`.
  New options: `first_chunk_frames`, `chunk_frames`, `max_seq_len`.
- `RussianLocalTTSService.synthesize_stream()`: a hook for services that produce audio
  in pieces. The default yields `synthesize()` as one chunk.

## [0.1.0] - 2026-11-15

### Added
- `GigaAMSTTService`: GigaAM v3 (`v3_e2e_rnnt` by default) as a Pipecat segmented STT
  service. It feeds tensors directly (no temporary files) and falls back to GigaAM's
  `transcribe(path)` if the model's internals differ. Utterances over 24 s are split at
  quiet points. When the `gigaam` package cannot download a checkpoint, the weights come
  from the Hugging Face mirror `ai-sage/GigaAM-v3` at a pinned commit (weights only, no
  remote code). The `gigaam` extra carries GigaAM's dependencies; GigaAM itself is
  installed with `--no-deps`, because its `onnxruntime==1.23.*` pin conflicts with
  pipecat-ai.
- `Qwen3TTSService`: Qwen3-TTS with voice cloning from a reference clip (the clone prompt
  is built once) or a CustomVoice preset speaker.
- `SileroTTSService`: Silero v5 Russian as a CPU fallback. The weights are CC BY-NC 4.0.
- `RussianTextNormalizer`: a lexicon, Russian pre-rules (abbreviations, `в 2024 г.`, time
  ranges, ages, versions), rutextnorm, and optional ruaccent stress marks (`+`, acute
  accent, or none). It is usable as a Pipecat text transform.
- `RussianTextSplitter` / `RussianTextAggregator`: a short first chunk per reply, then
  whole sentences; abbreviations, initials and decimals never split a chunk.
- `BackchannelUserTurnStartStrategy` and `russian_user_turn_strategies()`: short
  "ага/угу/да/так" while the bot speaks do not interrupt it; smart-turn v3.2 is the stop
  strategy.
- `pipecat-ru-bench`: WER/CER and latency on a folder of your own recordings, and time to
  first audio and RTF for the TTS engines.
- Unit tests with mocked models, plus opt-in GPU integration tests
  (`PIPECAT_RU_GPU_TESTS=1`), verified on an RTX 5090: all pass.
- `examples/make_samples.py`: the same phrases with every engine, to compare by ear.

[Unreleased]: https://github.com/mustikon/pipecat-ru/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mustikon/pipecat-ru/releases/tag/v0.1.0
