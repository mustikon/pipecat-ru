# pipecat-ru

[![CI](https://github.com/mustikon/pipecat-ru/actions/workflows/ci.yml/badge.svg)](https://github.com/mustikon/pipecat-ru/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pipecat-ru.svg)](https://pypi.org/project/pipecat-ru/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Русскоязычные плагины для голосового фреймворка [Pipecat](https://github.com/pipecat-ai/pipecat):
локальные STT и TTS, а также обработка текста и очерёдности реплик, без которых русский голосовой
ассистент звучит неестественно.**

[English version](README.md)

Встроенные сервисы и настройки Pipecat рассчитаны на английский. Из коробки русский бот читает
«15.03 в 14:30» по цифрам и молчит, пока не получит целое предложение. А стоит сказать «угу»,
как он замолкает на полуслове. Этот пакет исправляет всё это открытыми моделями, которые
работают на вашей видеокарте.

| Компонент | Что делает |
|---|---|
| `GigaAMSTTService` | Финальная русская расшифровка с пунктуацией на [GigaAM v3](https://github.com/salute-developers/GigaAM). На русском точнее Whisper и не галлюцинирует на тишине. |
| `Qwen3TTSService` | Синтез речи на [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) с клонированием голоса по короткой русской записи. |
| `SileroTTSService` | [Silero v5](https://github.com/snakers4/silero-models): быстрый резервный голос на CPU. **Веса только для некоммерческого использования (CC BY-NC 4.0).** |
| `RussianTextNormalizer` | Числа, даты, время, деньги, единицы и сокращения (`т.е.`, `ул.`, `в 2024 г.`) прописью. Словарь произношений и, по желанию, ударения через [ruaccent](https://github.com/Den4ikAI/ruaccent). |
| `RussianTextAggregator` | Делит текст на предложения и клаузы так, что **первый** кусок каждого ответа короткий и звук начинается раньше. |
| `BackchannelUserTurnStartStrategy` | Пока бот говорит, короткие «ага / угу / да / так» (меньше ~600 мс) его не перебивают, а «стоп» перебивает. Работает в паре со smart-turn v3.2, который понимает русский. |
| `pipecat-ru-bench` | WER и задержка на папке ваших записей. Время до первого звука для TTS. |

## Установка

```bash
pip install pipecat-ru                       # текстовые инструменты и стратегия реплик (без torch)
pip install "pipecat-ru[gigaam,qwen,silero,accent]"
pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
```

GigaAM v3 нет на PyPI (пакет `gigaam` там — старая v1), поэтому он ставится с GitHub и с
`--no-deps`: GigaAM фиксирует `onnxruntime==1.23.*`, а pipecat-ai требует `onnxruntime>=1.24.3`.
onnxruntime нужен GigaAM только для экспорта в ONNX, остальные зависимости ставит extra `gigaam`. Файловому
API GigaAM нужен `ffmpeg` в `PATH`. Этот пакет передаёт модели тензоры напрямую, так что `ffmpeg`
нужен только как запасной путь.

**RTX 50xx (Blackwell) на Windows:** сначала поставьте PyTorch со сборкой CUDA 12.8+, например
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`. Старые колёса
не поддерживают `sm_120`. `flash-attn` необязателен. Передавайте
`attn_implementation="flash_attention_2"` в `Qwen3TTSService`, только если он установлен.

> `qwen-tts` фиксирует `transformers==4.57.3`. Не ставьте в то же окружение
> `pipecat-ai[local-smart-turn]`, ему нужен transformers 5.x. Он и не нужен: smart-turn v3.2
> работает на ONNX-рантайме из базовых зависимостей pipecat-ai.

## Быстрый старт

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

Полный бот от микрофона до колонок лежит в [`examples/local_bot.py`](examples/local_bot.py). Он
использует локальную OpenAI-совместимую LLM, а Silero подменяет Qwen3-TTS при сбоях.

## Компоненты

### STT на GigaAM

`GigaAMSTTService` — это `SegmentedSTTService` из Pipecat: VAD режет речь на фразы, каждая
распознаётся в рабочем потоке. Модель по умолчанию `v3_e2e_rnnt` выдаёт текст с пунктуацией и
нормализацией. `v3_e2e_ctc` быстрее. Фразы длиннее 24 с режутся по паузам, потому что короткий
API GigaAM ограничен 25 с. Можно передать готовую модель через `model=` и делить её между
сервисами. Иначе модель загружается при старте пайплайна.

Обычно веса скачиваются с CDN SberDevices через пакет `gigaam`. Если загрузка не удалась (в ноябре
2026 у CDN был просроченный TLS-сертификат), загрузчик берёт официальное зеркало на Hugging Face,
`ai-sage/GigaAM-v3`. Скачиваются только веса, конфиг и токенизатор с закреплённого коммита, чужой
код не запускается. Источник можно выбрать явно: `GigaAMSTTService(source="package" | "hf" | "auto")`.

### TTS: Qwen3-TTS и Silero

Оба сервиса синтезируют по одному куску текста за вызов в рабочем потоке и отдают звук кадрами
по 200 мс.

* **Клон голоса Qwen3-TTS:** `Qwen3TTSService(ref_audio=..., ref_text=...)` на модели *Base*
  1.7B или 0.6B. Промпт клона строится один раз при старте.
* **Готовый голос Qwen3-TTS:** `Qwen3TTSService(speaker="Ryan", instruct="...")` на модели
  *CustomVoice*.
* **Silero:** `SileroTTSService(speaker="xenia")`, голоса `aidar`, `baya`, `kseniya`, `xenia`,
  `eugene`. Работает на CPU и понимает ударения со знаком `+`.

Пакет `qwen-tts` пока не отдаёт звук по частям. Поэтому время до первого звука зависит от длины
первого куска, и держать его коротким — задача сплиттера.

Оба сервиса по умолчанию подключают русский сплиттер и нормализатор: `normalizer=None` /
`splitter=None` отключают их, можно передать и свои.

### Нормализатор текста

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

Шаги идут по порядку:

1. Словарь названий, которые TTS исказит (`Claude Code` → `Клод Код`, `GitHub` → `Гитхаб`).
   Его можно дополнить или отключить.
2. Русские правила для того, что упускает rutextnorm: сокращения, `в 2024 г.`, интервалы
   времени в родительном падеже, возраст, `версия 3.12`.
3. [rutextnorm](https://pypi.org/project/rutextnorm/): числа, даты, деньги и единицы с
   согласованием по падежам.
4. По желанию — ударения ruaccent: знаком `+` (для Silero), комбинируемым акутом или без знаков.

Чтобы подключить нормализатор к любому TTS в Pipecat, передайте
`text_transforms=[("*", normalizer.transform)]`.

### Быстрый первый кусок

`RussianTextSplitter` / `RussianTextAggregator` отдают первый кусок ответа на первой `,` `;` `:`
или тире, как только в нём набралось 3 слова. Дальше они отдают целые предложения, а предложение
длиннее 220 символов режут по границе клаузы. Сокращения, инициалы и десятичные дроби кусок не
разрывают. Настройка — через `SplitterConfig`.

```python
RussianTextSplitter().split("Хорошо, сейчас посмотрю, что там с погодой. Завтра ясно.")
# ['Хорошо, сейчас посмотрю,', 'что там с погодой.', 'Завтра ясно.']
```

### «Ага» не перебивает

`russian_user_turn_strategies()` возвращает `UserTurnStrategies` для Pipecat. Стратегия начала
реплики — `BackchannelUserTurnStartStrategy`, стратегия конца — smart-turn v3.2
(`LocalSmartTurnAnalyzerV3`).

Стратегия начала заменяет стандартные стратегии Pipecat по VAD и по расшифровке:

* **Бот молчит:** реплика начинается с началом речи, как раньше.
* **Бот говорит:** фраза перебивает, только если она длиннее `max_backchannel_secs` (0,6 с) или в
  расшифровке есть слово не из списка поддакиваний. Короткое «ага» выбрасывается из контекста, и
  бот продолжает говорить.

В `vad_stop_secs` укажите `stop_secs` вашего VAD. Список слов меняется через `backchannels=`.

### Бенчмарк на своём голосе

```bash
# папка: 001.wav + 001.txt (эталон), ... или transcripts.tsv "файл<TAB>текст"
pipecat-ru-bench stt my_phrases/ --device cuda --csv stt.csv
pipecat-ru-bench tts --engine qwen3 --ref-audio me.wav --ref-text "..." --save-dir out/
pipecat-ru-bench tts --engine silero
```

Для STT выводятся WER и CER (числа перед подсчётом прописываются с обеих сторон), задержка p50,
p90 и max и RTF. Для TTS — время до первого звука и RTF.

## Измеренная производительность

Замеры на RTX 5090 (32 ГБ), Windows 11, torch 2.11+cu128, bf16. Цифры взяты из
`tests/test_gpu_integration.py` и `examples/make_samples.py`.

| Шаг | Результат |
|---|---|
| GigaAM v3 e2e-RNNT, фраза 4 с | **66 мс**, расшифровка без ошибок |
| Нормализатор (без ударений) | 0,25 мс на предложение |
| Ударения ruaccent (`turbo3.1`, CPU) | 12–60 мс на кусок |
| Silero v5 (CPU), первый кусок | **60–140 мс**; ответ на 10 с готов за 0,25 с |
| Qwen3-TTS 1.7B (`qwen-tts` 0.1.1, sdpa) | **RTF ≈ 1,4**: 3 с на 2 с звука; первый звук через 4–10 с |

Через официальный пакет `qwen-tts` модель Qwen3-TTS на этой видеокарте **медленнее реального
времени**. Во время генерации GPU загружен примерно на 8%: узкое место — накладные расходы Python
на каждом шаге генерации. Модель 0.6B не быстрее. Для живых ответов пока используйте Silero (или
другой быстрый голос). Qwen3-TTS станет пригоден с более быстрым рантаймом, например vLLM-Omni
или декодером на CUDA graphs; это в планах.

ruaccent правильно ставит ударения в частых омографах («повесил зам+ок», «насыпь мук+у»). В
неоднозначных фразах («замок на воротах сломан») он может ошибиться. Ударения ставятся на каждый
кусок отдельно, так что модель видит меньше контекста, чем во всём ответе.

## Лицензии моделей

Код в этом репозитории под Apache-2.0. **У моделей свои лицензии**, и вы принимаете их, когда
скачиваете веса.

| Модель / данные | Где используется | Лицензия | Коммерческое использование |
|---|---|---|---|
| GigaAM v3 (SberDevices) | `GigaAMSTTService` | MIT | да |
| Qwen3-TTS (Alibaba Qwen) | `Qwen3TTSService` | Apache-2.0 | да |
| Silero TTS v5 | `SileroTTSService` | **CC BY-NC 4.0** | **нет** (нужна лицензия авторов) |
| ruaccent | `StressMarker` (по желанию) | MIT (по README проекта) | да |
| rutextnorm | `RussianTextNormalizer` | MIT | да |
| smart-turn v3.2 (Pipecat) | `russian_user_turn_strategies` | BSD-2-Clause | да |
| Silero VAD | `SileroVADAnalyzer` из Pipecat | MIT | да |

Лицензии меняются. Перед выпуском продукта проверьте карточку каждой модели.

## Тесты

```bash
uv sync
uv run pytest                    # все модели подменены: без GPU и скачиваний
```

Необязательные тесты на настоящих моделях для машины с CUDA (написаны под RTX 5090 и Windows 11):

```powershell
uv sync --all-extras
uv pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM"
$env:PIPECAT_RU_GPU_TESTS = "1"
$env:PIPECAT_RU_REF_AUDIO = "C:\voice\me.wav"         # по желанию, для клона голоса
$env:PIPECAT_RU_REF_TEXT  = "Текст этой записи."
uv run pytest -m gpu -s
```

## Статус и планы

Альфа-версия. Пакет сделан для личного ассистента и выложен на случай, если пригодится другим.
В планах:

* потоковое STT с промежуточными гипотезами (T-one или NVIDIA Nemotron streaming ASR), чтобы
  поддакивания и перебивания распознавались до конца фразы;
* рантайм Qwen3-TTS, работающий в реальном времени (vLLM-Omni или CUDA graphs), с потоковым звуком.

См. [CONTRIBUTING.md](CONTRIBUTING.md) и [CHANGELOG.md](CHANGELOG.md).
