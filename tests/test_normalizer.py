from __future__ import annotations

import pytest

from pipecat_ru.text import (
    RussianTextNormalizer,
    StressFormat,
    StressMarker,
    convert_plus_marks,
)
from tests.conftest import FakeAccentizer


@pytest.fixture
def norm() -> RussianTextNormalizer:
    return RussianTextNormalizer()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Мне 21 год.", "Мне двадцать один год."),
        ("Работаем с 9:00 до 18:00.", "Работаем с девяти до восемнадцати."),
        ("Открыто с 9 до 21:30.", "Открыто с девяти до двадцати одного тридцать."),
        ("В 2024 г. было тепло.", "В две тысячи двадцать четвёртом году было тепло."),
        ("Подожди 10 сек.", "Подожди десять секунд."),
        ("Запусти Claude Code и проверь GitHub.", "Запусти Клод Код и проверь Гитхаб."),
        ("Еще раз.", "Ещё раз."),
        ("Это так, т. к. он прав.", "Это так, так как он прав."),
        ("См. выше.", "Смотри выше."),
    ],
)
def test_examples(norm: RussianTextNormalizer, text: str, expected: str) -> None:
    assert norm(text) == expected


def test_numbers_dates_and_abbreviations(norm: RussianTextNormalizer) -> None:
    out = norm("Встреча 15.03.2025 в 14:30, т.е. через 2 дня.")
    assert "пятнадцатое марта две тысячи двадцать пятого года" in out
    assert "то есть" in out
    assert "два дня" in out
    assert not any(c.isdigit() for c in out)


def test_address(norm: RussianTextNormalizer) -> None:
    out = norm("ул. Ленина, д. 7, кв. 12")
    assert out.startswith("улица Ленина, дом семь, квартира двенадцать")


def test_centimetres_are_not_see(norm: RussianTextNormalizer) -> None:
    assert "сантиметр" in norm("Высота 5 см.")
    assert "смотри" not in norm("Высота 5 см.")


def test_versions_use_tochka(norm: RussianTextNormalizer) -> None:
    out = norm("Вышел Python 3.13.1, а версия 3.12 устарела.")
    assert "три точка тринадцать точка один" in out
    assert "три точка двенадцать" in out


def test_dates_are_not_versions(norm: RussianTextNormalizer) -> None:
    assert "точка" not in norm("Релиз 01.10.2024.")


def test_people_count(norm: RussianTextNormalizer) -> None:
    assert norm("Пришло 5 чел.") == "Пришло пять человек."


def test_custom_lexicon_overrides_default() -> None:
    n = RussianTextNormalizer(lexicon={"Claude": "Клауд", "Мустикон": "Мустик+он"})
    assert n("Claude и Мустикон") == "Клауд и Мустик+он"


def test_default_lexicon_can_be_disabled() -> None:
    n = RussianTextNormalizer(use_default_lexicon=False, use_rutextnorm=False)
    assert n("GitHub") == "GitHub"


def test_lexicon_is_whole_word_and_case_insensitive() -> None:
    n = RussianTextNormalizer(use_rutextnorm=False)
    assert n("github и GITHUB") == "Гитхаб и Гитхаб"
    assert n("GitHubber") == "GitHubber"


def test_empty_and_blank_text(norm: RussianTextNormalizer) -> None:
    assert norm("") == ""
    assert norm("   ") == "   "


def test_trailing_punctuation_is_kept(norm: RussianTextNormalizer) -> None:
    assert norm("Сколько стоит 1 кг?").endswith("?")
    assert norm("Всё готово!").endswith("!")


def test_stress_marks_plus(fake_accentizer: FakeAccentizer) -> None:
    n = RussianTextNormalizer(stress=StressMarker("plus", accentizer=fake_accentizer))
    assert n("Это замок.") == "Это з+амок."
    assert fake_accentizer.calls == ["Это замок."]


def test_stress_marks_acute(fake_accentizer: FakeAccentizer) -> None:
    n = RussianTextNormalizer(stress=StressMarker(StressFormat.ACUTE, accentizer=fake_accentizer))
    assert n("Это замок.") == "Это за́мок."


def test_stress_marks_none_strips(fake_accentizer: FakeAccentizer) -> None:
    n = RussianTextNormalizer(stress=StressMarker("none", accentizer=fake_accentizer))
    assert n("Это замок.") == "Это замок."


def test_convert_plus_marks_leaves_other_plus_signs() -> None:
    assert convert_plus_marks("2+2 и з+амок", StressFormat.ACUTE) == "2+2 и за́мок"


async def test_transform_matches_call(norm: RussianTextNormalizer) -> None:
    assert await norm.transform("Мне 21 год.", "sentence") == norm("Мне 21 год.")


async def test_transform_runs_stress_in_a_thread(fake_accentizer: FakeAccentizer) -> None:
    n = RussianTextNormalizer(stress=StressMarker(accentizer=fake_accentizer))
    assert await n.transform("привет") == "прив+ет"


def test_stress_marker_without_ruaccent_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "ruaccent":
            raise ModuleNotFoundError("No module named 'ruaccent'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pipecat-ru\[accent\]"):
        StressMarker().load()
