# MO Rag Evaluation — оценка RAG через LLM-as-a-Judge

Как пользоваться — **[USAGE.md](USAGE.md)**. Развёртывание на сервере — **[DEPLOY.md](DEPLOY.md)**.

## Что это

Закрытая (по паролю) веб-арена для оценки RAG-системы **МОРАГ**: загружаешь JSONL с ответами RAG →
LLM-судья сравнивает каждый ответ с эталонным (gold) → строка в лидерборде с accuracy.
Плюс в UI можно руками править эталоны (с версиями) и руками переопределять вердикты судьи.

Gradio-приложение (один процесс `app.py`), без БД — всё в файлах.

## Структура

### Файлы кода

| Файл | Роль |
|---|---|
| `app.py` | Весь UI: вход по паролю, лидерборд, сабмит, детали по вопросам, правка gold, human override, версии |
| `src/envs.py` | Конфиг: пароли, модель судьи, пути |
| `src/submission/submit.py` | **Судья**: промпт, вызов LLM, кэш, логика dual-judge/ToCheck, подсчёт ссылок |
| `src/gold_versioning.py` | Черновик/фиксация версий gold, diff |
| `utils/md_to_qa.py` | Отдельная утилита: генерация вопросов-ответов из markdown (Grok) |
| `requirements.lock.txt` | Точные версии из venv прода (gradio 6.10, openai 2.30) |

### Данные (не в репозитории — приватные, только на сервере)

| Путь | Что |
|---|---|
| `data/gold/answers_gold.jsonl` | **Активный эталон**, 396 вопросов: `{id, question, answer, context}` |
| `data/gold/answers_gold_old.jsonl` | Есть только пока идёт черновик правок |
| `data/gold/versions/` | Архив прошлых версий + `*.diff.md` что поменялось |
| `data/questions/questions_public.jsonl` | Публичные вопросы (без ответов) — пересобирается из gold при старте |
| `data/human_overrides.json` | Ручные вердикты: `{метка сабмита: {qid: correct/wrong/to_check}}` |
| `data/judge_cache.json` | Кэш вердиктов судьи |
| `eval_details.jsonl` | Все прогоны с вердиктами по каждому вопросу (21 МБ) |
| `leaderboard.csv` | Лидерборд |

⚠️ `data/gold/` и `human_overrides.json` — то, что правит человек: бэкапьте их.

## Как работает оценка

1. Пользователь загружает JSONL: `{"id": "0", "answer": "...", "sources": ["https://...", ...]}`
   (`sources` необязательно; нужно для метрики LinksOK — совпали ли ссылки с `context` эталона).
2. Выбирает версию gold (active или архивную) и, опционально, галку «Secondary judge».
3. В фоне по каждому вопросу (5 параллельно) судья отвечает строго `{"score": 0|1}`.
4. Итог по вопросу:
   - один судья → его вердикт;
   - два судьи согласны → Correct / Wrong; разошлись или ошибка → **ToCheck**.
5. **Accuracy = Correct / (ответов − ToCheck)** — ToCheck исключается из знаменателя.
6. Пустой ответ = Wrong без вызова судьи.

## Судьи — где править

Всё в **`src/submission/submit.py`** + переменные окружения в systemd-юните.

| Что поменять | Где |
|---|---|
| Модель основного судьи | env `EVAL_MODEL` (сейчас `deepseek/deepseek-v4-flash`, через OpenRouter). Дефолт — `src/envs.py` |
| Модель второго судьи | env `SECONDARY_MODEL`; если пусто, при галке в UI берётся `DEFAULT_SECONDARY_MODEL = "qwen/qwen3.5-flash-02-23"` в `submit.py` |
| Промпт судьи | `SYSTEM_PROMPT` и `USER_PROMPT_TEMPLATE` в `submit.py` |
| Сбросить кэш после смены промпта | **поднять `_PROMPT_VERSION`** (сейчас `"v3"`) в `submit.py`. Иначе старые вердикты из кэша будут переиспользованы |
| Параллельность | env `EVAL_CONCURRENCY` (по умолчанию 5) |
| Ключ API | env `OPENROUTER_API_KEY` (оба судьи ходят через OpenRouter). `XAI_API_KEY` — наследие, не используется |
| Прокси до OpenRouter | env `PROXY_URL` (сейчас не задан) |
| Подпись судьи на странице | текст в `app.py` (блок «Dataset info» и шапка) — **руками**, сам не обновляется |

Ключ кэша: `qid + версия промпта + md5(ответ) + md5(эталон)`. Значит: тот же ответ на тот же эталон
не перепроверяется; правка эталона по вопросу автоматически даёт новую оценку. **Смена модели кэш
не сбрасывает** — при смене модели тоже поднимайте `_PROMPT_VERSION`.

Применить изменения:

```bash
cp src/submission/submit.py src/submission/submit.py.bak-$(date +%Y%m%d)
# правка submit.py или env в systemd-юните (systemctl edit <service>)
systemctl daemon-reload && systemctl restart <service>
journalctl -u <service> -f   # [judge:...] ошибки, [eval] cached/new
```

## Что правит человек (всё через UI, после входа по паролю)

### 1. Эталоны (gold) — секция «🔍 Evaluation details + ✏️ Edit gold»
- У каждого вопроса кнопка **✏️ Edit** → правится вопрос, эталонный ответ, источник (context) → **💾 Save**.
- Первая правка переводит gold в режим **черновика** (старая версия отложена в `_old`).
  Правки сразу действуют на новые сабмиты против `active`.
- **💾 Save New Gold Set** — фиксирует версию: старая уходит в `versions/` с diff.
  Секция «📚 Gold versions» — список версий и просмотр diff.
- При сабмите можно выбрать, против какой версии судить.

### 2. Вердикты судьи (human override)
- Выбрать сабмит в «Select submission» → у каждого вопроса кнопки 🟢 Correct / 🔴 Wrong / 🟡 ToCheck
  (повторный клик снимает). Хранится в `data/human_overrides.json`, число правок видно в лидерборде.
- **📨 Submit Human Overrides** — создаёт новую строку лидерборда `<имя>_HumChg` с пересчитанной accuracy.
  Исходная строка остаётся.

### 3. Сабмиты
- Загрузка JSONL в «📤 Submit», потом **🔄 Refresh**. Скачать вопросы — «📥 Downloads».

## Пароль на вход
`EDIT_PASSWORDS` (через запятую) в env. В коде стоит заглушка **`pass`** — **обязательно смените**, задав `EDIT_PASSWORDS`.

## Быстрый локальный запуск

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.lock.txt
export OPENROUTER_API_KEY=... EVAL_MODEL=deepseek/deepseek-v4-flash EDIT_PASSWORDS=...
export GOLD_PATH_LOCAL=$PWD/data/gold/answers_gold.jsonl SHOW_DETAILS=1
venv/bin/python app.py      # http://localhost:7862
```

Нужен `data/gold/answers_gold.jsonl` (не в репо — приватный). Формат строки:
`{"id": "1", "question": "...", "answer": "...", "context": "Источник: https://..."}`

## Грабли
- `data/questions/questions_public.jsonl` в репо — заглушка; на сервере пересобирается из gold при старте.
- `utils/md_to_qa.py` использует Grok (xAI) — отдельная утилита, к судье не относится.
- Gold местами сам ошибочен — при спорных Wrong сначала проверить эталон, а не RAG.
