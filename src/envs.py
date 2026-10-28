import os
import json

# ── Auth ─────────────────────────────────────────────────────────────────────
# Пароли для входа на страницу (csv) — задаются через env EDIT_PASSWORDS.
# Дефолт "pass" — заглушка: ОБЯЗАТЕЛЬНО СМЕНИТЬ (задать EDIT_PASSWORDS в окружении).
EDIT_PASSWORDS = {
    p.strip()
    for p in os.getenv("EDIT_PASSWORDS", "pass").split(",")
    if p.strip()
}

# ── Primary judge через OpenRouter: deepseek-v4.1-flash (с 28.10.26; до того v4-flash с 17.06.26, ещё раньше Grok/xAI) ──
XAI_API_KEY = os.getenv("XAI_API_KEY", "")  # legacy
EVAL_MODEL = os.getenv("EVAL_MODEL", "deepseek/deepseek-v4.1-flash")
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "5"))
PROXY_URL = os.getenv("PROXY_URL", "")

# ── Пути к данным ─────────────────────────────────────────────────────────────
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "data/questions/questions_public.jsonl")

# Папка с gold-файлами и архивом версий.
# Внутри:
#   answers_gold.jsonl         — active (может быть в режиме черновика)
#   answers_gold_old.jsonl     — присутствует только пока active=draft
#   versions/                  — архив зафиксированных версий + diffs
GOLD_DIR = os.getenv("GOLD_DIR", "data/gold")
GOLD_ACTIVE_FILE = "answers_gold.jsonl"
GOLD_OLD_FILE = "answers_gold_old.jsonl"
GOLD_VERSIONS_SUBDIR = "versions"

# Эталонные ответы — параметры загрузки
_HF_TOKEN = os.getenv("HF_TOKEN", "")
_GOLD_DATASET_ID = os.getenv("GOLD_DATASET_ID", "datakomarov/RAG-data-v2")
_GOLD_FILENAME = os.getenv("GOLD_FILENAME", "answers_gold.jsonl")

# Ленивая загрузка — вызывается только при первом evaluate_submission
_gold_path_cache = None

def get_gold_path() -> str:
    """Возвращает путь к gold-файлу, загружая его при первом вызове."""
    global _gold_path_cache
    if _gold_path_cache is not None:
        return _gold_path_cache

    local_override = os.getenv("GOLD_PATH_LOCAL", "")
    if local_override and os.path.exists(local_override):
        _gold_path_cache = local_override
        return _gold_path_cache

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_GOLD_DATASET_ID,
            filename=_GOLD_FILENAME,
            repo_type="dataset",
            token=_HF_TOKEN,
            local_dir=".",
        )
        _gold_path_cache = path
        return _gold_path_cache
    except Exception as e:
        raise RuntimeError(
            f"Cannot load gold answers from '{_GOLD_DATASET_ID}/{_GOLD_FILENAME}'. "
            f"Set GOLD_PATH_LOCAL env var to use a local file. Error: {e}"
        )


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# ── OpenRouter / Secondary judge ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
SECONDARY_MODEL = os.getenv('SECONDARY_MODEL', '')
JUDGE_CACHE_PATH = os.getenv('JUDGE_CACHE_PATH', 'data/judge_cache.json')
