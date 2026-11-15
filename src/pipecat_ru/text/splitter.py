"""Russian sentence/clause splitter that makes the first TTS chunk short.

Pipecat's default aggregator waits for a whole sentence before the TTS starts. For a
voice assistant the time to the *first* audio is what the user feels, so this splitter
releases the first chunk of every bot reply at the first clause boundary (``,`` ``;``
``:`` or a dash) once it holds a few words, then goes back to whole sentences, cutting
an over-long sentence at a clause boundary.

Russian abbreviations (``т.е.``, ``т.к.``, ``и т.д.``, ``ул.``, ``г.``), initials
(``А. С. Пушкин``) and decimals (``7,5`` or ``3.14``) never split a chunk.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator, TextType

SENTENCE_END = ".!?…"
CLAUSE_END = ",;:"
DASHES = "—–-"
CLOSERS = "\"'»”)]"
OPENERS = "\"'«„“([—–-"

# Lowercased abbreviations (without the final dot) after which a dot does not end
# a sentence, even when a capital letter follows ("ул. Ленина", "г. Москва").
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # multi-part: the word before the final dot, e.g. "т.е" -> "е" is caught below
        "т.е",
        "т.к",
        "т.н",
        "т.п",
        "т.д",
        "т.ч",
        "и.о",
        "н.э",
        "до н.э",
        "с.г",
        # single-word
        "г",
        "гг",
        "в",
        "вв",
        "ул",
        "пр",
        "просп",
        "пер",
        "пл",
        "д",
        "кв",
        "корп",
        "стр",
        "р",
        "руб",
        "коп",
        "тыс",
        "млн",
        "млрд",
        "трлн",
        "долл",
        "чел",
        "шт",
        "см",
        "ср",
        "им",
        "проф",
        "акад",
        "доц",
        "др",
        "мин",
        "сек",
        "ч",
        "оз",
        "обл",
        "р-н",
        "пос",
        "дер",
        "с",
        "ст",
        "напр",
        "англ",
        "рус",
        "лат",
        "т",
        "е",
        "к",
        "н",
        "п",
        "рис",
        "табл",
        "гл",
        "пп",
        "т.о",
        "ок",
        "прим",
        "ред",
        "изд",
        "сб",
        "вып",
        "янв",
        "фев",
        "февр",
        "мар",
        "апр",
        "авг",
        "сент",
        "окт",
        "нояб",
        "дек",
        "тел",
        "моб",
        "eg",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "mr",
        "mrs",
        "dr",
    }
)

_WORD_RE = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)


def count_words(text: str) -> int:
    """Count words the way a listener would (hyphenated words count once)."""
    return len(_WORD_RE.findall(text))


@dataclass(frozen=True)
class SplitterConfig:
    """Tuning for :class:`RussianTextSplitter`.

    Parameters:
        first_chunk_min_words: The first chunk of a reply is released at the first
            clause or sentence boundary that leaves at least this many words in it.
        first_chunk_max_words: If set, the first chunk is cut at a word boundary once
            it reaches this many words even without punctuation. Off by default,
            because a cut in the middle of a clause sounds unnatural.
        max_chunk_chars: Later chunks are whole sentences, but a sentence longer than
            this is cut at its last clause boundary so the TTS never waits for a
            paragraph-long sentence.
    """

    first_chunk_min_words: int = 3
    first_chunk_max_words: int | None = None
    max_chunk_chars: int = 220


class RussianTextSplitter:
    """Incremental, synchronous splitter: feed text, get finished chunks back.

    This is the engine behind :class:`RussianTextAggregator`; use it directly in
    scripts or tests.

    Example::

        splitter = RussianTextSplitter()
        splitter.split("Хорошо, сейчас посмотрю. Погода в Москве ясная.")
        # ['Хорошо, сейчас посмотрю.', 'Погода в Москве ясная.']
    """

    def __init__(self, config: SplitterConfig | None = None) -> None:
        """Create a splitter with the given (or default) tuning."""
        self.config = config or SplitterConfig()
        self._buf = ""
        self._first = True

    @property
    def pending(self) -> str:
        """Text received but not released yet."""
        return self._buf

    def reset(self) -> None:
        """Forget buffered text; the next chunk counts as the first of a reply."""
        self._buf = ""
        self._first = True

    def feed(self, text: str) -> list[str]:
        """Add streamed text and return the chunks that are now complete."""
        out: list[str] = []
        for ch in text:
            self._buf += ch
            chunk = self._try_release()
            while chunk is not None:
                out.append(chunk)
                chunk = self._try_release()
        return out

    def flush(self) -> str | None:
        """End of a reply: return what is left and reset for the next reply."""
        rest = self._buf.strip()
        self.reset()
        return rest or None

    def split(self, text: str) -> list[str]:
        """Split a whole text at once (one reply)."""
        chunks = self.feed(text)
        rest = self.flush()
        if rest:
            chunks.append(rest)
        return chunks

    # ------------------------------------------------------------------ internals

    def _try_release(self) -> str | None:
        cut = self._find_cut()
        if cut is None:
            return None
        chunk = self._buf[:cut].strip()
        self._buf = self._buf[cut:].lstrip()
        if not chunk:
            return None
        self._first = False
        return chunk

    def _find_cut(self) -> int | None:
        """Return the buffer offset to cut at, or None to keep buffering."""
        buf = self._buf
        # A boundary is confirmed by the first non-space character after it, so a
        # buffer that ends in whitespace cannot hold a new confirmed boundary.
        if not buf or buf[-1].isspace():
            return None
        boundaries = self._boundaries(buf)

        if self._first:
            for cut, _ in boundaries:
                if count_words(buf[:cut]) >= self.config.first_chunk_min_words:
                    return cut
            limit = self.config.first_chunk_max_words
            if limit is not None:
                # cut before the word that is being typed now
                last_space = buf.rfind(" ")
                if last_space > 0 and count_words(buf[:last_space]) >= limit:
                    return last_space
            return None

        for cut, is_sentence in boundaries:
            if is_sentence:
                return cut
        if len(buf) > self.config.max_chunk_chars:
            clause_cuts = [cut for cut, _ in boundaries if count_words(buf[:cut]) >= 2]
            if clause_cuts:
                return clause_cuts[-1]
        return None

    def _boundaries(self, buf: str) -> list[tuple[int, bool]]:
        """List (cut offset, is_sentence_end) for every confirmed boundary in buf."""
        result: list[tuple[int, bool]] = []
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch in SENTENCE_END or ch in CLAUSE_END:
                j = i
                while j + 1 < n and (buf[j + 1] in SENTENCE_END or buf[j + 1] in CLAUSE_END):
                    j += 1
                cut = j + 1
                while cut < n and buf[cut] in CLOSERS:
                    cut += 1
                m = cut
                while m < n and buf[m].isspace():
                    m += 1
                # needs whitespace after the punctuation and a lookahead character
                if cut < m < n:
                    kind = self._classify(buf, i, buf[i : j + 1], buf[m])
                    if kind is not None:
                        result.append((cut, kind))
                i = j + 1
                continue
            if ch in DASHES and 0 < i < n - 1 and buf[i - 1] == " " and buf[i + 1] == " ":
                # " — " clause break: cut before the dash so it opens the next chunk
                m = i + 1
                while m < n and buf[m].isspace():
                    m += 1
                if m < n and count_words(buf[:i]) > 0:
                    result.append((i, False))
            i += 1
        return result

    def _classify(self, buf: str, start: int, punct: str, lookahead: str) -> bool | None:
        """Classify punctuation as a sentence end (True), clause break (False) or none."""
        if not any(c in SENTENCE_END for c in punct):
            return False
        if punct != ".":
            return True  # "!", "?", "...", "?!" and the like
        if self._is_abbreviation_dot(buf, start):
            # an initial ("А. Пушкин") or abbreviation ("ул. Ленина") never splits;
            # the price is that "... и т.д. Потом" stays in one chunk
            return None
        # a dot followed by a lowercase word is a clause break at most
        return lookahead.isupper() or lookahead.isdigit() or lookahead in OPENERS

    @staticmethod
    def _word_before(buf: str, dot: int) -> str:
        start = dot
        while start > 0 and (buf[start - 1].isalnum() or buf[start - 1] in ".-"):
            start -= 1
        return buf[start:dot]

    def _is_abbreviation_dot(self, buf: str, dot: int) -> bool:
        word = self._word_before(buf, dot).lower().strip(".")
        if not word:
            return False
        if word in ABBREVIATIONS:
            return True
        # "и т. д." written with spaces: the last part is a lone short letter
        return len(word) == 1 and word.isalpha()


class RussianTextAggregator(BaseTextAggregator):
    """Pipecat text aggregator built on :class:`RussianTextSplitter`.

    Every chunk is released as a :data:`TextType.SENTENCE` aggregation, so text
    transforms registered for sentences (such as the Russian normalizer) still apply.
    The TTS services in this package install it automatically; to use it with any
    other Pipecat TTS, pass it to
    :class:`~pipecat.processors.aggregators.llm_text_processor.LLMTextProcessor`.
    """

    def __init__(self, config: SplitterConfig | None = None) -> None:
        """Create the aggregator (language is fixed to Russian)."""
        super().__init__(aggregation_type=TextType.SENTENCE, language="ru")
        self._splitter = RussianTextSplitter(config)

    @property
    def text(self) -> Aggregation:
        """Text buffered but not released yet."""
        return Aggregation(text=self._splitter.pending.strip(), type=TextType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Feed streamed LLM text and yield each finished chunk."""
        for chunk in self._splitter.feed(text):
            yield Aggregation(text=chunk, type=TextType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        """Release what is left at the end of the reply."""
        rest = self._splitter.flush()
        return Aggregation(text=rest, type=TextType.SENTENCE) if rest else None

    async def handle_interruption(self) -> None:
        """Drop buffered text when the user interrupts."""
        self._splitter.reset()

    async def reset(self) -> None:
        """Drop buffered text and start a new reply."""
        self._splitter.reset()
