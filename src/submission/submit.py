# src/submission/submit.py  — dual LLM-as-judge: Grok (primary) + Qwen via OpenRouter (secondary)
# Judgments are cached by (qid, pred_hash) in data/judge_cache.json
import json
import re
import hashlib
import threading
import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.envs import (
    XAI_API_KEY,
    EVAL_MODEL,
    EVAL_CONCURRENCY,
    PROXY_URL,
    QUESTIONS_PATH,
    get_gold_path,
    load_jsonl,
    OPENROUTER_API_KEY,
    SECONDARY_MODEL,
    JUDGE_CACHE_PATH,
)
from src.gold_versioning import gold_path_for_version

SECONDARY_ENABLED = bool(SECONDARY_MODEL and OPENROUTER_API_KEY)
DEFAULT_SECONDARY_MODEL = "qwen/qwen3.5-flash-02-23"  # используется если юзер в UI включил secondary, но SECONDARY_MODEL env пуст


# ── Метрика «ссылка попала в gold» ───────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+")


def _normalize_url(u: str) -> str:
    u = (u or "").strip().rstrip(".,;)'\"!? ")
    u = u.split("#")[0]
    return u


def _extract_urls(text: str) -> set:
    if not text:
        return set()
    return {_normalize_url(u) for u in _URL_RE.findall(text)}


def _link_score(d: dict) -> int:
    """1 если хоть одна ссылка модели совпала с ссылкой из gold.context, иначе 0."""
    gold_urls = _extract_urls(d.get("gold_context", "") or "")
    pred_urls = set()
    for s in d.get("sources") or []:
        if isinstance(s, str):
            pred_urls |= _extract_urls(s)
        elif isinstance(s, dict) and s.get("url"):
            pred_urls |= _extract_urls(str(s["url"]))
    if not gold_urls or not pred_urls:
        return 0
    return 1 if (gold_urls & pred_urls) else 0

# ── Прогресс ─────────────────────────────────────────────────────────────────
_progress: dict = {}  # filename -> {"done", "total", "cached", "pt": primary total time, "pn": primary count, "st", "sn"}
_progress_lock = threading.Lock()

def get_progress(filename: str) -> dict:
    with _progress_lock:
        return _progress.get(filename, {})

# ── Кэш вердиктов ─────────────────────────────────────────────────────────────
# Структура: {key: {"primary": int|None, "secondary": int|None, "score": int, "to_check": bool}}
_cache: dict = {}
_cache_lock = threading.Lock()

def _load_cache():
    global _cache
    try:
        with open(JUDGE_CACHE_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
        print(f"[cache] loaded {len(_cache)} entries from {JUDGE_CACHE_PATH}")
    except FileNotFoundError:
        _cache = {}
    except Exception as e:
        print(f"[cache] failed to load: {e}")
        _cache = {}

def _save_cache():
    try:
        with open(JUDGE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)
    except Exception as e:
        print(f"[cache] failed to save: {e}")

# Bump при изменении SYSTEM_PROMPT — это инвалидирует кэш судьи (старые записи
# в judge_cache.json остаются, но не используются: новые ключи имеют новую версию).
_PROMPT_VERSION = "v3"


def _cache_key(qid: str, pred: str, gold: str) -> str:
    ph = hashlib.md5(pred.encode("utf-8")).hexdigest()[:8]
    gh = hashlib.md5(gold.encode("utf-8")).hexdigest()[:8]
    return f"{qid}:{_PROMPT_VERSION}:{ph}{gh}"

_cache_loaded = False

def _ensure_cache():
    global _cache_loaded
    if not _cache_loaded:
        _load_cache()
        _cache_loaded = True


# ── Клиенты ──────────────────────────────────────────────────────────────────
_client_primary = None
_client_secondary = None

def _get_primary():
    global _client_primary
    if _client_primary is None:
        http_client = httpx.Client(
            proxy=PROXY_URL if PROXY_URL else None,
            timeout=httpx.Timeout(3600.0),
        )
        # Primary judge переведён с Grok (xAI) на deepseek-v4-flash через OpenRouter (17.06.26)
        _client_primary = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            http_client=http_client,
        )
    return _client_primary

def _get_secondary():
    global _client_secondary
    if _client_secondary is None:
        http_client = httpx.Client(
            proxy=PROXY_URL if PROXY_URL else None,
            timeout=httpx.Timeout(60.0),
        )
        _client_secondary = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            http_client=http_client,
        )
    return _client_secondary


# ── Промпты ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Ты — оценщик ответов RAG-системы.
Твоя задача: сравнить ответ RAG-системы с эталонным ответом и решить, правильный он или нет.

Ты ДОЛЖЕН выбрать одно из двух — частичных баллов не бывает.

Правила оценки:
  1 — ПРАВИЛЬНО. Ответ RAG содержит все ключевые факты из эталона
      (семантически эквивалентно, перефразировано или как НАДМНОЖЕСТВО эталона).
      ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ разрешена и НЕ делает ответ неправильным:
      детали, контекст, примеры, структура (списки, таблицы, заголовки),
      цитирование источников, расширение сверх эталона — всё это допустимо,
      если каждый ключевой факт из эталона присутствует и не противоречит ему.
  0 — НЕПРАВИЛЬНО. В ответе отсутствует один или несколько ключевых фактов
      из эталона, ответ противоречит эталону, содержит неверную информацию,
      пуст или нерелевантен вопросу.

Важно: НЕ штрафуй за длину, формат, стиль или наличие дополнительной правильной
информации сверх эталона. Штрафуй только за пропущенные или неверные ключевые факты.

В ответе верни ТОЛЬКО валидный JSON-объект и ничего больше.
Формат: {"score": 0|1}
"""

USER_PROMPT_TEMPLATE = """\
Вопрос:
{question}

Эталонный ответ:
{gold}

Ответ RAG-системы:
{pred}

Ответь в формате json.
"""


def _parse_score(text: str) -> int:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return -1
    try:
        obj = json.loads(m.group(0))
        s = int(obj.get("score", -1))
        return 1 if s == 1 else 0
    except Exception:
        return -1


# Структура JSON-вывода: строгий enum [0, 1].
_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "Judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}


def _judge(
    client,
    model: str,
    question: str,
    gold: str,
    pred: str,
    disable_reasoning: bool = False,
) -> tuple[int, float]:
    import time as _time
    prompt = USER_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)
    t0 = _time.time()
    base_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    if disable_reasoning:
        # OpenRouter spec: для моделей с thinking-режимом явно его выключаем.
        # У моделей без поддержки — параметр просто игнорируется.
        base_kwargs["extra_body"] = {"reasoning": {"enabled": False, "exclude": True}}

    # 1-я попытка — со строгой json_schema. Если провайдер модели её не
    # поддерживает (некоторые модели через OpenRouter) — fallback на json_object.
    for response_format in (_JUDGE_SCHEMA, {"type": "json_object"}):
        try:
            resp = client.chat.completions.create(response_format=response_format, **base_kwargs)
            score = _parse_score(resp.choices[0].message.content)
            elapsed = round(_time.time() - t0, 2)
            return score, elapsed
        except Exception as e:
            msg = str(e)
            # Если ошибка про response_format / json_schema — пробуем fallback.
            if response_format is _JUDGE_SCHEMA and any(
                kw in msg.lower() for kw in ("response_format", "json_schema", "schema", "structured")
            ):
                print(f"[judge:{model}] json_schema unsupported, fallback to json_object: {e}")
                continue
            elapsed = round(_time.time() - t0, 2)
            print(f"[judge:{model}] error in {elapsed}s: {e}")
            return -1, elapsed
    return -1, round(_time.time() - t0, 2)


def _eval_one(
    qid: str,
    question: str,
    gold: str,
    pred: str,
    use_secondary: bool = False,
    secondary_model: str = "",
    sources: list | None = None,
    gold_version: str = "active",
) -> dict:
    pred = (pred or "").strip()
    sources = sources or []
    if not pred:
        return {
            "id": qid, "question": question, "gold": gold, "pred": pred,
            "sources": sources,
            "score_primary": 0, "score_secondary": 0,
            "score": 0, "to_check": False, "cached": False,
        }

    key = _cache_key(qid, pred, gold)

    with _cache_lock:
        entry = dict(_cache.get(key, {}))

    # Взять primary из кэша или запросить (не кэшируем -1)
    cached_primary = entry.get("primary")
    if cached_primary in (0, 1):
        score_primary, pt = cached_primary, 0.0
        primary_cached = True
    else:
        score_primary, pt = _judge(_get_primary(), EVAL_MODEL, question, gold, pred, disable_reasoning=False)
        primary_cached = False
        if score_primary in (0, 1):  # кэшируем только валидные ответы
            with _cache_lock:
                _cache.setdefault(key, {})["primary"] = score_primary
                _save_cache()

    # Secondary — только если включён юзером; иначе имитируем «согласие» для финальной развилки
    if use_secondary and OPENROUTER_API_KEY and secondary_model:
        cached_secondary = entry.get("secondary")
        if cached_secondary in (0, 1):
            score_secondary, st = cached_secondary, 0.0
            secondary_cached = True
        else:
            score_secondary, st = _judge(_get_secondary(), secondary_model, question, gold, pred, disable_reasoning=True)
            secondary_cached = False
            if score_secondary in (0, 1):  # кэшируем только валидные ответы
                with _cache_lock:
                    _cache.setdefault(key, {})["secondary"] = score_secondary
                    _save_cache()
    else:
        score_secondary = score_primary
        st = 0.0
        secondary_cached = True

    # Обновить статистику времени в прогрессе
    if not primary_cached or not secondary_cached:
        with _progress_lock:
            for fname, p in _progress.items():
                if not primary_cached:
                    p["pt"] = p.get("pt", 0) + pt
                    p["pn"] = p.get("pn", 0) + 1
                if not secondary_cached:
                    p["st"] = p.get("st", 0) + st
                    p["sn"] = p.get("sn", 0) + 1

    fully_cached = primary_cached and secondary_cached

    if score_primary == -1 or score_secondary == -1:
        to_check = True
        score = 0
    elif score_primary == score_secondary:
        to_check = False
        score = score_primary
    else:
        to_check = True
        score = 0

    with _cache_lock:
        _cache.setdefault(key, {}).update({"score": score, "to_check": to_check})
        # Audit: какие версии gold выдавали этот же вердикт (gold-контент идентичен).
        versions = _cache[key].setdefault("gold_versions", [])
        if gold_version and gold_version not in versions:
            versions.append(gold_version)
        _save_cache()

    return {
        "id": qid, "question": question, "gold": gold, "pred": pred,
        "sources": sources,
        "score_primary": score_primary,
        "score_secondary": score_secondary,
        "score": score,
        "to_check": to_check,
        "cached": fully_cached,
    }


def evaluate_submission(
    submit_path: str,
    use_secondary: bool = False,
    secondary_model: str = "",
    gold_version: str = "active",
) -> dict:
    _ensure_cache()
    # Fallback: если пользователь включил secondary но модель не указана — берём дефолт
    if use_secondary and not secondary_model:
        secondary_model = SECONDARY_MODEL or DEFAULT_SECONDARY_MODEL

    sub_rows = load_jsonl(submit_path)
    pred_map = {str(x["id"]): str(x.get("answer", "")).strip() for x in sub_rows}
    sources_map = {str(x["id"]): x.get("sources") or [] for x in sub_rows}

    if gold_version == "active":
        gold_path = get_gold_path()
    else:
        gold_path = gold_path_for_version(gold_version)
    gold_rows = load_jsonl(gold_path)

    gold_map = {}
    question_map = {}
    gold_context_map = {}
    for x in gold_rows:
        xid = str(x["id"])
        gold_map[xid] = str(x.get("answer", ""))
        if "question" in x:
            question_map[xid] = x["question"]
        if "context" in x:
            gold_context_map[xid] = x["context"]

    try:
        pub_questions = load_jsonl(QUESTIONS_PATH)
        for q in pub_questions:
            qid = str(q["id"])
            if qid not in question_map:
                question_map[qid] = q.get("question", "")
    except Exception:
        pass

    total = len(gold_map)
    answered_ids = [qid for qid in gold_map if pred_map.get(qid, "")]

    details = []
    _filename = submit_path.split("/")[-1]
    with _progress_lock:
        _progress[_filename] = {"done": 0, "total": len(answered_ids), "cached": 0, "pt": 0, "pn": 0, "st": 0, "sn": 0}

    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as executor:
        futures = {
            executor.submit(
                _eval_one,
                qid,
                question_map.get(qid, ""),
                gold_map[qid],
                pred_map[qid],
                use_secondary,
                secondary_model,
                sources_map.get(qid, []),
                gold_version,
            ): qid
            for qid in answered_ids
        }
        for future in as_completed(futures):
            try:
                d = future.result()
                d["gold_context"] = gold_context_map.get(d.get("id", ""), "")
                d["links_ok"] = _link_score(d)
                details.append(d)
            except Exception as e:
                qid = futures[future]
                print(f"[judge] future error on {qid}: {e}")
                d = {"id": qid, "score": 0, "to_check": True, "cached": False, "links_ok": 0}
                details.append(d)
            with _progress_lock:
                _progress[_filename]["done"] += 1
                if d.get("cached"):
                    _progress[_filename]["cached"] += 1

    cached_count = sum(1 for d in details if d.get("cached"))
    to_check_count = sum(1 for d in details if d.get("to_check"))
    correct = sum(1 for d in details if d.get("score") == 1 and not d.get("to_check"))
    wrong = sum(1 for d in details if d.get("score") == 0 and not d.get("to_check"))
    n_valid = len(answered_ids) - to_check_count

    print(f"[eval] cached: {cached_count}, new: {len(answered_ids) - cached_count}")

    links_ok_total = sum(int(d.get("links_ok") or 0) for d in details)

    return {
        "zeros": wrong,
        "ones": correct,
        "to_check": to_check_count,
        "n_submit": len(answered_ids),
        "n_valid": n_valid,
        "total": total,
        "details": details,
        "gold_version": gold_version,
        "links_ok": links_ok_total,
    }
