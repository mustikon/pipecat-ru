# Contributing

Thanks for helping! The most useful reports for a Russian speech package are concrete:
the text that was normalized or split wrongly, or the phrase that was misrecognized,
together with what you expected.

## Development setup

```bash
git clone https://github.com/mustikon/pipecat-ru && cd pipecat-ru
uv sync
uv run pytest
uv run ruff check src tests examples && uv run ruff format --check src tests examples
uv run mypy
```

The unit tests never load a real model: GigaAM, Qwen3-TTS, Silero and ruaccent are
replaced with the doubles in `tests/conftest.py`, so the suite runs on any machine in
seconds. Real-model tests live in `tests/test_gpu_integration.py` and run only with
`PIPECAT_RU_GPU_TESTS=1` (see the README).

## Normalizer and splitter fixes

Add the failing sentence to `tests/test_normalizer.py` or `tests/test_splitter.py` first.
Prefer a narrow rule in `pipecat_ru/text/normalizer.py` over a broad one: a rule that
fixes one phrase and breaks ten others is worse than no rule. Problems inside rutextnorm
itself belong upstream (https://github.com/shigabeev/russian_tts_normalization).

## Pull requests

* Keep the core light: model runtimes go into extras, never into `dependencies`.
* Check the current Pipecat API before touching a service. It changes often, and the
  supported range is pinned in `pyproject.toml`.
* Add a `CHANGELOG.md` entry under *Unreleased*.
* CI must be green on Linux, macOS and Windows.
