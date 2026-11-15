from __future__ import annotations

import pytest

from pipecat_ru.text import RussianTextAggregator, RussianTextSplitter, SplitterConfig, count_words


def split(text: str, **cfg: object) -> list[str]:
    return RussianTextSplitter(SplitterConfig(**cfg)).split(text)  # type: ignore[arg-type]


def test_first_chunk_is_released_at_first_clause() -> None:
    assert split("Хорошо, сейчас посмотрю, что там с погодой. Завтра ясно.") == [
        "Хорошо, сейчас посмотрю,",
        "что там с погодой.",
        "Завтра ясно.",
    ]


def test_first_chunk_waits_for_min_words() -> None:
    # "Да." alone is shorter than 3 words, so it joins the next clause
    assert split("Да. Сейчас посмотрю, что там.")[0] == "Да. Сейчас посмотрю,"


def test_min_words_one_releases_immediately() -> None:
    assert split("Да. Сейчас посмотрю, что там.", first_chunk_min_words=1)[0] == "Да."


def test_later_chunks_are_whole_sentences() -> None:
    chunks = split("Ну что ж, начнём. Первый пункт, второй пункт, третий пункт. Конец!")
    # "Ну что ж," already has 3 words, so it is the short first chunk
    assert chunks == ["Ну что ж,", "начнём.", "Первый пункт, второй пункт, третий пункт.", "Конец!"]


@pytest.mark.parametrize(
    "text",
    [
        "Это было в 1799 г. в Москве, и т.д. и т.п. так и было.",
        "Он живёт на ул. Ленина, т. е. в центре.",
        "Пушкин, А. С., родился давно.",
    ],
)
def test_abbreviations_and_initials_do_not_split_sentences(text: str) -> None:
    chunks = split(text, first_chunk_min_words=100)
    assert chunks == [text]


def test_initials_do_not_split_after_first_chunk() -> None:
    chunks = split(
        "Начнём, пожалуй. А. С. Пушкин родился в Москве. Это всё.", first_chunk_min_words=1
    )
    assert "А. С. Пушкин родился в Москве." in chunks


def test_decimals_are_not_boundaries() -> None:
    chunks = split("Курс 92,5 рубля и рост 3.14 процента сегодня. Дальше.")
    assert chunks[0] == "Курс 92,5 рубля и рост 3.14 процента сегодня."


def test_dash_is_a_clause_break_before_the_dash() -> None:
    assert split("Москва — столица России и большой город.", first_chunk_min_words=1) == [
        "Москва",
        "— столица России и большой город.",
    ]


def test_question_and_exclamation_end_sentences() -> None:
    assert split("Ты уверен в этом? Да! Точно.", first_chunk_min_words=1) == [
        "Ты уверен в этом?",
        "Да!",
        "Точно.",
    ]


def test_closing_quotes_stay_with_their_sentence() -> None:
    chunks = split("Он сказал: «Я приду завтра.» Потом ушёл.", first_chunk_min_words=1)
    assert chunks == ["Он сказал:", "«Я приду завтра.»", "Потом ушёл."]


def test_long_sentence_is_cut_at_last_clause() -> None:
    long = "Сначала скажу вот что, " + "очень длинное перечисление, " * 12 + "и конец."
    chunks = split(long, max_chunk_chars=120)
    assert len(chunks) > 2
    assert all(len(c) <= 160 for c in chunks)
    assert " ".join(chunks) == long.strip()


def test_first_chunk_max_words_forces_a_cut() -> None:
    text = "раз два три четыре пять шесть семь восемь девять десять"
    chunks = split(text, first_chunk_max_words=4)
    assert chunks[0] == "раз два три четыре"
    assert " ".join(chunks) == text


def test_streaming_token_by_token_matches_whole_text() -> None:
    text = "Хорошо, сейчас посмотрю. Погода в Москве ясная, ветер слабый. Всё!"
    splitter = RussianTextSplitter()
    streamed: list[str] = []
    for i in range(0, len(text), 3):
        streamed.extend(splitter.feed(text[i : i + 3]))
    rest = splitter.flush()
    if rest:
        streamed.append(rest)
    assert streamed == RussianTextSplitter().split(text)


def test_flush_resets_first_chunk_state() -> None:
    splitter = RussianTextSplitter()
    assert (
        splitter.split("Первый ответ, длинный и подробный. Да.")[0]
        == "Первый ответ, длинный и подробный."
    )
    # second reply: short first chunk again
    assert splitter.split("Второй ответ тоже, с запятой. Ага.")[0] == "Второй ответ тоже,"


def test_count_words() -> None:
    assert count_words("северо-западный ветер, 7 м/с") == 5


async def test_aggregator_interface() -> None:
    agg = RussianTextAggregator()
    out = [a.text async for a in agg.aggregate("Хорошо, сейчас посмотрю. Погода ")]
    assert out == ["Хорошо, сейчас посмотрю."]
    assert agg.text.text == "Погода"
    out = [a.text async for a in agg.aggregate("ясная. Ветер")]
    assert out == ["Погода ясная."]
    rest = await agg.flush()
    assert rest is not None and rest.text == "Ветер"
    assert await agg.flush() is None


async def test_aggregator_interruption_resets_buffer() -> None:
    agg = RussianTextAggregator()
    _ = [a async for a in agg.aggregate("Начинаю длинный ответ")]
    await agg.handle_interruption()
    assert agg.text.text == ""
    out = [a.text async for a in agg.aggregate("Новый ответ, короткий. ")]
    assert out == []  # needs lookahead
    out = [a.text async for a in agg.aggregate("Да")]
    assert out == ["Новый ответ, короткий."]
