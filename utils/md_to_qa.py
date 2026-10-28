#!/usr/bin/env python3
"""
Читает ВСЕ .md файлы из текущей директории,
для каждого генерирует вопросы, ответы и контекст через Grok API
с использованием Structured Outputs (response_format с JSON Schema),
и сохраняет результат в .jsonl файл с тем же именем.

Большие файлы (> CHUNK_THRESHOLD символов) автоматически бьются на чанки.

Использование:
    python md_to_qa.py

Требования:
    pip install openai

Переменная окружения:
    XAI_API_KEY — ваш API ключ от xAI (Grok)
"""

import os
import json
import glob
import time

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────
GROK_MODEL         = "grok-4-1-fast-non-reasoning"
CHARS_PER_QUESTION  = 1000  # ~1 вопрос на 1000 символов
MIN_QUESTIONS_TOTAL = 1     # минимум вопросов на документ
MAX_QUESTIONS_CHUNK = 50    # максимум вопросов за один запрос (один чанк)
CONTEXT_MAX_TOKENS  = 512
REQUEST_TIMEOUT    = 600

MAX_RETRIES        = 5
RETRY_BASE_DELAY   = 5
RATE_LIMIT_DELAY   = 15

CHUNK_THRESHOLD    = 50_000
CHUNK_SIZE         = 40_000

# ──────────────────────────────────────────────
# JSON Schema для structured output
# ──────────────────────────────────────────────
QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Вопрос для RAG-оценки"
                            },
                            "answer": {
                                "type": "string",
                                "description": "Развёрнутый ответ на основе текста"
                            },
                            "context": {
                                "type": "string",
                                "description": "Дословная цитата из текста с ответом"
                            }
                        },
                        "required": ["question", "answer", "context"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
    }
}


def get_api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Переменная окружения XAI_API_KEY не задана.\n"
            "Задайте её: export XAI_API_KEY='your_key_here'"
        )
    return key


def find_all_md(directory: str) -> list[str]:
    pattern = os.path.join(directory, "*.md")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Нет .md файлов в директории: {directory}")
    return files


def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Бьёт текст на куски по chunk_size символов, стараясь резать по абзацам."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind("\n\n", start, end)
        if cut <= start:
            cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return [c.strip() for c in chunks if c.strip()]


def call_with_retry(client: OpenAI, **kwargs) -> object:
    """Запрос к API с повторами при временных ошибках."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"\n   ⚠️  Rate limit (попытка {attempt}/{MAX_RETRIES}), ждём {RATE_LIMIT_DELAY} сек...")
            time.sleep(RATE_LIMIT_DELAY)

        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"\n   ⚠️  {type(e).__name__} (попытка {attempt}/{MAX_RETRIES}), ждём {delay} сек...")
            time.sleep(delay)


def get_doc_identity(client: OpenAI, text: str) -> str:
    """Шаг 1: кратко описываем документ для контекста при формулировке вопросов."""
    prompt = f"""Прочитай текст ниже и ответь одним абзацем (3-5 предложений):
- Кто автор (если указан)?
- Какова тема работы?
- Какой тип документа (курсовая, магистерская диссертация, научная статья, учебная программа и т.п.)?
- Какие ключевые методы, модели или результаты упоминаются?

Ответ нужен для формулировки вопросов вида
"В работе [автор] по теме [тема] — какой метод был использован для X?"

ТЕКСТ (первые 3000 символов):
{text[:3000]}
"""
    response = call_with_retry(
        client,
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=120,
    )
    return response.choices[0].message.content.strip()


def generate_qa_for_chunk(client: OpenAI, chunk: str, doc_identity: str, num_questions: int) -> list[dict]:
    """Генерирует Q&A для одного куска текста через structured output."""
    prompt = f"""Ты — эксперт по созданию датасетов для оценки RAG-систем.

ОПИСАНИЕ ДОКУМЕНТА (используй для формулировки вопросов):
{doc_identity}

ЗАДАЧА: по тексту ниже сгенерируй ровно {num_questions} вопросов с ответами.

ТРЕБОВАНИЯ К ВОПРОСАМ:
1. Вопрос должен идентифицировать работу через автора, тему или контекст —
   без указания на документ или его структуру.
   Плохо: "Какой ROC AUC у LightGBM?"  ← непонятно в какой работе
   Плохо: "Что написано в разделе 3?"  ← ссылка на структуру
   Хорошо: "Какой ROC AUC показал LightGBM в работе Фролова по кредитному скорингу Сбербанка?"
   Хорошо: "Какую архитектуру нейросети использовал Фролов для обработки транзакций заемщиков?"

2. Вопросы должны охватывать разные аспекты: методологию, данные, результаты, выводы.

3. Избегай тривиальных вопросов "что такое X?" для общеизвестных терминов.

Поле "context" — дословная цитата из текста, где содержится ответ (максимум {CONTEXT_MAX_TOKENS} токенов).

ТЕКСТ:
{chunk}
"""

    response = call_with_retry(
        client,
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format=QA_SCHEMA,
        timeout=REQUEST_TIMEOUT,
    )

    data = json.loads(response.choices[0].message.content)
    return data["items"]


def process_file(client: OpenAI, md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    char_count = len(text)
    total_questions = max(MIN_QUESTIONS_TOTAL, char_count // CHARS_PER_QUESTION)
    print(f"   Символов: {char_count} → вопросов: {total_questions}")

    # Шаг 1: описание документа
    print(f"   Шаг 1: определяем тип и контекст документа...", end=" ", flush=True)
    doc_identity = get_doc_identity(client, text)
    print("✓")
    print(f"   → {doc_identity[:120].replace(chr(10), ' ')}...")

    # Шаг 2: генерация Q&A
    use_chunks = char_count > CHUNK_THRESHOLD
    all_qa = []

    if not use_chunks:
        n_q = min(total_questions, MAX_QUESTIONS_CHUNK)
        print(f"   Шаг 2: генерируем {n_q} вопросов...", end=" ", flush=True)
        t0 = time.time()
        all_qa = generate_qa_for_chunk(client, text, doc_identity, n_q)
        print(f"✓ получено пар: {len(all_qa)} ({time.time() - t0:.1f} сек)")
    else:
        chunks = split_into_chunks(text, CHUNK_SIZE)
        print(f"   Шаг 2: файл большой, разбит на {len(chunks)} чанков")

        total_chars = sum(len(c) for c in chunks)
        distributed = 0
        for i, chunk in enumerate(chunks, 1):
            if i == len(chunks):
                n_q = total_questions - distributed
            else:
                n_q = max(1, round(total_questions * len(chunk) / total_chars))
            n_q = min(n_q, MAX_QUESTIONS_CHUNK)  # не больше лимита за один запрос
            distributed += n_q

            print(f"   Чанк {i}/{len(chunks)} ({len(chunk)} симв.) → {n_q} вопросов...", end=" ", flush=True)
            t0 = time.time()
            try:
                qa = generate_qa_for_chunk(client, chunk, doc_identity, n_q)
                all_qa.extend(qa)
                print(f"✓ получено {len(qa)} ({time.time() - t0:.1f} сек)")
            except Exception as e:
                print(f"❌ пропущен: {e}")

    print(f"   Итого пар: {len(all_qa)}")

    base_name = os.path.splitext(md_path)[0]
    out_path = base_name + ".jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return out_path


def main():
    directory = os.path.abspath(".")
    print(f"📂 Директория: {directory}")
    print(f"🤖 Модель: {GROK_MODEL}\n")

    md_files = find_all_md(directory)
    print(f"📋 Найдено .md файлов: {len(md_files)}\n")

    client = OpenAI(
        api_key=get_api_key(),
        base_url="https://api.x.ai/v1",
    )

    errors = []
    for i, md_path in enumerate(md_files, 1):
        filename = os.path.basename(md_path)
        print(f"[{i}/{len(md_files)}] 📄 {filename}")
        try:
            out_path = process_file(client, md_path)
            print(f"   💾 Сохранено: {os.path.basename(out_path)}\n")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}\n")
            errors.append((filename, str(e)))

    print("─" * 40)
    print(f"✅ Обработано: {len(md_files) - len(errors)}/{len(md_files)}")
    if errors:
        print("❌ Ошибки:")
        for fname, err in errors:
            print(f"   {fname}: {err}")


if __name__ == "__main__":
    main()
