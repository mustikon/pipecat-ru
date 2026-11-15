"""Stress marks for Russian TTS via `ruaccent <https://github.com/Den4ikAI/ruaccent>`_.

ruaccent writes ``+`` before the stressed vowel (``з+амок``). Silero reads that
form natively; other models may prefer a combining acute accent after the vowel
(``за́мок``) or no marks at all, so the output format is configurable.
"""

from __future__ import annotations

import re
import threading
from enum import StrEnum
from typing import Any, Protocol

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
COMBINING_ACUTE = "́"
_PLUS_VOWEL_RE = re.compile(rf"\+([{VOWELS}])")


class StressFormat(StrEnum):
    """How stress marks are written into the text sent to the TTS."""

    PLUS = "plus"  # "з+амок" (Silero, ruaccent's native form)
    ACUTE = "acute"  # "за́мок" (combining U+0301 after the vowel)
    NONE = "none"  # strip marks; still useful for ruaccent's "ё" restoration


class Accentizer(Protocol):
    """The part of ``ruaccent.RUAccent`` this module uses."""

    def process_all(self, text: str) -> str:
        """Return text with ``+`` before every stressed vowel."""
        ...


def convert_plus_marks(text: str, fmt: StressFormat) -> str:
    """Convert ruaccent's ``+vowel`` marks into the requested format."""
    if fmt is StressFormat.PLUS:
        return text
    if fmt is StressFormat.ACUTE:
        return _PLUS_VOWEL_RE.sub(lambda m: m.group(1) + COMBINING_ACUTE, text)
    return _PLUS_VOWEL_RE.sub(lambda m: m.group(1), text)


class StressMarker:
    """Add stress marks with ruaccent, loading the model lazily on first use.

    Args:
        fmt: Output format for the marks.
        model_size: ruaccent homograph model (``"turbo3.1"``, ``"turbo2"``, ``"tiny2.1"``...).
        use_dictionary: Use ruaccent's dictionary (more accurate, more RAM).
        device: ``"CPU"`` or ``"CUDA"``.
        accentizer: A ready accentizer (an ``RUAccent`` instance or a test double);
            skips loading ruaccent.
        load_kwargs: Extra keyword arguments for ``RUAccent.load``.
    """

    def __init__(
        self,
        fmt: StressFormat | str = StressFormat.PLUS,
        *,
        model_size: str = "turbo3.1",
        use_dictionary: bool = True,
        device: str = "CPU",
        accentizer: Accentizer | None = None,
        load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Configure the marker; ruaccent is imported only when first needed."""
        self.fmt = StressFormat(fmt)
        self._model_size = model_size
        self._use_dictionary = use_dictionary
        self._device = device
        self._load_kwargs = load_kwargs or {}
        self._accentizer = accentizer
        self._lock = threading.Lock()

    def load(self) -> Accentizer:
        """Load ruaccent now (call it at startup to avoid a first-reply delay)."""
        with self._lock:
            if self._accentizer is None:
                try:
                    from ruaccent import RUAccent
                except ModuleNotFoundError as e:  # pragma: no cover - depends on extras
                    raise ImportError(
                        'Stress marks need ruaccent: pip install "pipecat-ru[accent]"'
                    ) from e
                acc = RUAccent()
                acc.load(
                    omograph_model_size=self._model_size,
                    use_dictionary=self._use_dictionary,
                    device=self._device,
                    **self._load_kwargs,
                )
                self._accentizer = acc
            return self._accentizer

    def __call__(self, text: str) -> str:
        """Return ``text`` with stress marks in the configured format."""
        if not text.strip():
            return text
        marked = self.load().process_all(text)
        return convert_plus_marks(marked, self.fmt)
