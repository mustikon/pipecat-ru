"""Russian text normalizer for TTS.

Turns written Russian into something a synthesizer can say. The pipeline, in order:

1. **Lexicon**: whole-word, case-insensitive replacements for names the model would
   mispronounce (``Claude`` -> ``Клод``). A small default set covers words a
   personal assistant says often; add your own or switch the defaults off.
2. **Pre-rules**: Russian abbreviations and patterns that
   `rutextnorm <https://pypi.org/project/rutextnorm/>`_ leaves alone or gets wrong
   (``т.к.``, ``ул.``, ``в 2024 г.``, ``с 9:00 до 18:00``, ``Мне 21 год``, versions).
3. **rutextnorm**: numbers, dates, money, units, percentages, times, ordinals,
   Latin transliteration, all in grammatical agreement.
4. **Stress** (optional): ``+`` or acute marks from ruaccent via :class:`StressMarker`.

Use it as a plain function, or as a Pipecat text transform::

    normalizer = RussianTextNormalizer()
    tts = AnyPipecatTTS(..., text_transforms=[("*", normalizer.transform)])
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from typing import Any

import rutextnorm

from pipecat_ru.text.stress import StressMarker

# Words a Russian voice assistant meets daily that rule-based transliteration
# mangles ("GitHub" -> "гизуб"). Keys are matched case-insensitively as whole words.
DEFAULT_LEXICON: dict[str, str] = {
    "Claude Code": "Клод Код",
    "Claude": "Клод",
    "Anthropic": "Антропик",
    "OpenAI": "Опен Эй Ай",
    "ChatGPT": "Чат Джи Пи Ти",
    "Codex": "Кодекс",
    "Gemini": "Джемини",
    "Antigravity": "Антигравити",
    "GitHub": "Гитхаб",
    "Google": "Гугл",
    "YouTube": "Ютуб",
    "Telegram": "Телеграм",
    "WhatsApp": "Ватсап",
    "Windows": "Виндоус",
    "Linux": "Линукс",
    "iPhone": "Айфон",
    "Apple": "Эпл",
    "Python": "Пайтон",
    "Pipecat": "Пайпкэт",
    "NVIDIA": "Энвидиа",
    "RTX": "Эр Ти Икс",
    "Wi-Fi": "Вай-Фай",
    "WiFi": "Вай-Фай",
    "email": "имейл",
    "e-mail": "имейл",
    "OK": "окей",
}

Replacement = str | Callable[[re.Match[str]], str]

# (pattern, replacement). Applied in order, before rutextnorm.
_SPACE = r"\s?"
_PRE_RULES: list[tuple[re.Pattern[str], Replacement]] = [
    (re.compile(rf"\bт\.{_SPACE}е\.", re.IGNORECASE), "то есть"),
    (re.compile(rf"\bт\.{_SPACE}к\.", re.IGNORECASE), "так как"),
    (re.compile(rf"\bт\.{_SPACE}н\.", re.IGNORECASE), "так называемый"),
    (re.compile(rf"\bв\s+т\.{_SPACE}ч\.", re.IGNORECASE), "в том числе"),
    (re.compile(rf"\bи\s+т\.{_SPACE}д\.", re.IGNORECASE), "и так далее"),
    (re.compile(rf"\bи\s+т\.{_SPACE}п\.", re.IGNORECASE), "и тому подобное"),
    (re.compile(r"\bи\s+др\.", re.IGNORECASE), "и другие"),
    (re.compile(r"\bи\s+пр\.", re.IGNORECASE), "и прочее"),
    (re.compile(r"\bнапр\.", re.IGNORECASE), "например"),
    (
        re.compile(r"(?<!\d)(?<!\d\s)\bсм\.(?=\s|$)", re.IGNORECASE),
        lambda m: "Смотри" if m.group(0)[0].isupper() else "смотри",
    ),
    (re.compile(r"\bул\.\s*"), "улица "),
    (re.compile(r"\bпросп\.\s*|\bпр-т\b\.?\s*"), "проспект "),
    (re.compile(r"\bпер\.\s*(?=[А-ЯЁ])"), "переулок "),
    (re.compile(r"\bпл\.\s*(?=[А-ЯЁ])"), "площадь "),
    (re.compile(r"\bд\.\s*(?=\d)"), "дом "),
    (re.compile(r"\bкв\.\s*(?=\d)"), "квартира "),
    (re.compile(r"\bкорп\.\s*(?=\d)"), "корпус "),
    (re.compile(r"\bстр\.\s*(?=\d)"), "страница "),
    (re.compile(r"\bим\.\s*(?=[А-ЯЁ])"), "имени "),
    (re.compile(r"\bпроф\.\s*(?=[А-ЯЁ])"), "профессор "),
    (re.compile(r"\bг\.\s*(?=[А-ЯЁ][а-яё])"), "город "),
    (re.compile(r"(?<=\d)\s*чел\.?(?=[\s,.;:!?]|$)"), " человек"),
    (re.compile(r"(?<=\d)\s*шт\.?(?=[\s,.;:!?]|$)"), " штук"),
]

# "в 2024 г." -> "в 2024 году" (rutextnorm reads a bare "г." as nominative "год")
_IN_YEAR_RE = re.compile(r"\b([Вв])\s+(\d{3,4})\s*г\.?(?=[\s,.;:!?]|$)")
# "с 9:00 до 18:00" -> genitive hours
_TIME_RANGE_RE = re.compile(
    r"\b([Сс])\s+(\d{1,2})(?::(\d{2}))?\s+до\s+(\d{1,2})(?::(\d{2}))?(?!\d)"
)
# "Мне 21 год" -> cardinal; rutextnorm reads it as an ordinal year
_AGE_RE = re.compile(r"(?<!\b[Вв]\s)(?<!\d)\b(\d{1,3})\s+(год|года|лет)\b")
# "версия 3.12", "Python 3.13.1", "v2.4"
_VERSION_RE = re.compile(
    r"(?:(?<=\bверсия )|(?<=\bверсии )|(?<=\bверсию )|(?<=\bверсией )|(?<=\bv))"
    r"(\d+(?:\.\d+)+)\b"
    # a bare X.Y.Z, unless it looks like a date (01.10.2024, 1.2.24)
    r"|\b(?!\d{1,2}\.\d{1,2}\.\d{2}(?:\d{2})?\b)(\d+\.\d+\.\d+)\b(?!\.\d)",
    re.IGNORECASE,
)

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
SENTENCE_END = ".!?…"


def _hours_gen(hours: str, minutes: str | None) -> str:
    words = rutextnorm.number_to_words_case(int(hours), "gen")
    if minutes and int(minutes):
        words += " " + rutextnorm.number_to_words(int(minutes))
    return str(words)


def _time_range(m: re.Match[str]) -> str:
    prep, h1, m1, h2, m2 = m.groups()
    if int(h1) > 24 or int(h2) > 24:
        return m.group(0)
    return f"{prep} {_hours_gen(h1, m1)} до {_hours_gen(h2, m2)}"


def _age(m: re.Match[str]) -> str:
    return f"{rutextnorm.number_to_words(int(m.group(1)))} {m.group(2)}"


def _version(m: re.Match[str]) -> str:
    version = m.group(1) or m.group(2)
    return " точка ".join(version.split("."))


def _compile_lexicon(lexicon: Mapping[str, str]) -> re.Pattern[str] | None:
    if not lexicon:
        return None
    # longest first, so "Claude Code" wins over "Claude"
    keys = sorted(lexicon, key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in keys)
    return re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])", re.IGNORECASE)


class RussianTextNormalizer:
    """Normalize Russian text for TTS (see the module docstring for the steps).

    Args:
        lexicon: Extra whole-word replacements; they override the defaults.
        use_default_lexicon: Include :data:`DEFAULT_LEXICON`.
        stress: A :class:`StressMarker` to add stress marks, or None for no marks.
        use_rutextnorm: Run rutextnorm (switch off only to debug the other steps).
        restore_yo: Restore unambiguous ``ё`` (``еще`` -> ``ещё``) when no stress
            marker is set (ruaccent restores ``ё`` itself).
    """

    def __init__(
        self,
        *,
        lexicon: Mapping[str, str] | None = None,
        use_default_lexicon: bool = True,
        stress: StressMarker | None = None,
        use_rutextnorm: bool = True,
        restore_yo: bool = True,
    ) -> None:
        """Build the normalizer; nothing heavy is loaded here."""
        merged: dict[str, str] = dict(DEFAULT_LEXICON) if use_default_lexicon else {}
        merged.update(lexicon or {})
        self._lexicon = {k.lower(): v for k, v in merged.items()}
        self._lexicon_re = _compile_lexicon(merged)
        self.stress = stress
        self._use_rutextnorm = use_rutextnorm
        self._restore_yo = restore_yo

    def __call__(self, text: str) -> str:
        """Return ``text`` normalized for speech."""
        if not text.strip():
            return text
        out = self.apply_lexicon(text)
        out = self.apply_pre_rules(out)
        if self._use_rutextnorm:
            out = str(rutextnorm.normalize_russian(out))
        if self._restore_yo and self.stress is None:
            out = str(rutextnorm.restore_yo(out))
        out = self._tidy(out, original=text)
        if self.stress is not None:
            out = self.stress(out)
        return out

    async def transform(self, text: str, text_type: Any = None) -> str:
        """Pipecat ``TextTransform``: ``text_transforms=[("*", normalizer.transform)]``.

        Runs in a worker thread when stress marking (a neural model) is enabled.
        """
        if self.stress is not None:
            return await asyncio.to_thread(self, text)
        return self(text)

    def apply_lexicon(self, text: str) -> str:
        """Apply only the lexicon step."""
        if self._lexicon_re is None:
            return text
        return self._lexicon_re.sub(lambda m: self._lexicon[m.group(0).lower()], text)

    @staticmethod
    def apply_pre_rules(text: str) -> str:
        """Apply only the abbreviation and pattern rules."""
        out = _IN_YEAR_RE.sub(r"\1 \2 году", text)
        out = _TIME_RANGE_RE.sub(_time_range, out)
        out = _VERSION_RE.sub(_version, out)
        out = _AGE_RE.sub(_age, out)
        for pattern, repl in _PRE_RULES:
            out = pattern.sub(repl, out)
        return out

    @staticmethod
    def _tidy(text: str, *, original: str) -> str:
        out = _MULTISPACE_RE.sub(" ", text)
        out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", out).strip()
        # rutextnorm drops a sentence-final dot that belonged to an abbreviation
        # ("10 сек." -> "десять секунд"); the TTS needs it for falling intonation.
        last = original.rstrip()[-1:]
        if last in SENTENCE_END and out and out[-1] not in SENTENCE_END:
            out += last
        return out
