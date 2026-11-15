"""Russian text tools for TTS: normalizer, stress marks and a fast-first-chunk splitter."""

from pipecat_ru.text.normalizer import DEFAULT_LEXICON, RussianTextNormalizer
from pipecat_ru.text.splitter import (
    RussianTextAggregator,
    RussianTextSplitter,
    SplitterConfig,
    count_words,
)
from pipecat_ru.text.stress import StressFormat, StressMarker, convert_plus_marks

__all__ = [
    "DEFAULT_LEXICON",
    "RussianTextAggregator",
    "RussianTextNormalizer",
    "RussianTextSplitter",
    "SplitterConfig",
    "StressFormat",
    "StressMarker",
    "convert_plus_marks",
    "count_words",
]
