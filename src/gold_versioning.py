"""Gold versioning.

Жизненный цикл:

    1. Изначально на диске только active (`data/gold/answers_gold.jsonl`).
       is_draft() == False.
    2. Пользователь редактирует впервые (apply_edit) → автоматически копируем
       active в `answers_gold_old.jsonl`. is_draft() == True.
    3. Дальше любые правки идут в active. _old не трогаем — это снапшот
       последней зафиксированной версии.
    4. Кнопка "🔒 Fix version" переносит _old → versions/answers_gold_<ts>.jsonl,
       рядом кладёт <ts>.diff.md (что изменилось между _old и текущим active).
       is_draft() возвращается к False; active становится новой
       зафиксированной версией.

Submitter в форме сабмита может выбрать любую архивную версию (или active).
В кэше вердиктов ключ уже содержит md5(gold[qid]), так что между версиями,
где конкретный qid не правился, кэш переиспользуется.
"""
from __future__ import annotations
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.envs import (
    GOLD_DIR,
    GOLD_ACTIVE_FILE,
    GOLD_OLD_FILE,
    GOLD_VERSIONS_SUBDIR,
    QUESTIONS_PATH,
)


# ── Пути ──────────────────────────────────────────────────────────────────────

def _active_path() -> str:
    # GOLD_PATH_LOCAL — обратная совместимость с systemd-env, который указывает
    # на конкретный файл. Если задан — берём из него каталог как GOLD_DIR.
    override = os.getenv("GOLD_PATH_LOCAL", "")
    if override:
        return override
    return os.path.join(GOLD_DIR, GOLD_ACTIVE_FILE)


def _gold_dir() -> str:
    return os.path.dirname(_active_path()) or "."


def _old_path() -> str:
    return os.path.join(_gold_dir(), GOLD_OLD_FILE)


def _versions_dir() -> str:
    p = os.path.join(_gold_dir(), GOLD_VERSIONS_SUBDIR)
    os.makedirs(p, exist_ok=True)
    return p


# ── Состояние ─────────────────────────────────────────────────────────────────

def is_draft() -> bool:
    return os.path.exists(_old_path())


def list_versions() -> list[dict]:
    """Архивные версии, отсортированные по дате (свежие первыми)."""
    out = []
    vd = _versions_dir()
    for name in sorted(os.listdir(vd), reverse=True):
        if not name.startswith("answers_gold_") or not name.endswith(".jsonl"):
            continue
        ts = name[len("answers_gold_"):-len(".jsonl")]
        path = os.path.join(vd, name)
        try:
            row_count = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        except Exception:
            row_count = 0
        diff_path = os.path.join(vd, f"answers_gold_{ts}.diff.md")
        out.append({
            "ts": ts,
            "path": path,
            "rows": row_count,
            "diff_path": diff_path if os.path.exists(diff_path) else None,
        })
    return out


def gold_path_for_version(version: str) -> str:
    """version: 'active' либо timestamp вида '20260508_153000'."""
    if version == "active":
        return _active_path()
    candidate = os.path.join(_versions_dir(), f"answers_gold_{version}.jsonl")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Gold version not found: {version}")
    return candidate


def submit_dropdown_choices() -> list[tuple[str, str]]:
    """Опции для dropdown в форме сабмита: [(label, value)]."""
    label_active = "active (draft)" if is_draft() else "active"
    choices = [(label_active, "active")]
    for v in list_versions():
        choices.append((f"{v['ts']} ({v['rows']} qids)", v["ts"]))
    return choices


# ── Чтение/запись JSONL ───────────────────────────────────────────────────────

def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ── Правки ────────────────────────────────────────────────────────────────────

def _start_draft_if_needed() -> None:
    if not is_draft():
        shutil.copy2(_active_path(), _old_path())


def get_row(qid: str) -> dict | None:
    qid = str(qid)
    for r in _read_jsonl(_active_path()):
        if str(r.get("id")) == qid:
            return r
    return None


def apply_edit(
    qid: str,
    question: str | None,
    answer: str | None,
    context: str | None = None,
) -> dict:
    """Правит question/answer/context для qid. None = не менять. Стартует draft."""
    qid = str(qid)
    _start_draft_if_needed()

    rows = _read_jsonl(_active_path())
    target = None
    for r in rows:
        if str(r.get("id")) == qid:
            target = r
            if question is not None:
                r["question"] = question
            if answer is not None:
                r["answer"] = answer
            if context is not None:
                r["context"] = context
            break
    if target is None:
        raise ValueError(f"qid {qid!r} not found in active gold")
    _write_jsonl(_active_path(), rows)

    # Синхрон публичного questions_public.jsonl при правке вопроса —
    # иначе сабмиттер скачает старую формулировку.
    if question is not None and os.path.exists(QUESTIONS_PATH):
        qrows = _read_jsonl(QUESTIONS_PATH)
        for r in qrows:
            if str(r.get("id")) == qid:
                r["question"] = question
                break
        _write_jsonl(QUESTIONS_PATH, qrows)

    return target


def revert_qid(qid: str) -> dict | None:
    """Возврат qid к значению из _old. Возвращает восстановленный row или None."""
    qid = str(qid)
    if not is_draft():
        return None
    old_rows = _read_jsonl(_old_path())
    old_row = next((r for r in old_rows if str(r.get("id")) == qid), None)
    if not old_row:
        return None

    rows = _read_jsonl(_active_path())
    for i, r in enumerate(rows):
        if str(r.get("id")) == qid:
            rows[i] = dict(old_row)
            _write_jsonl(_active_path(), rows)
            if os.path.exists(QUESTIONS_PATH):
                qrows = _read_jsonl(QUESTIONS_PATH)
                for j, qr in enumerate(qrows):
                    if str(qr.get("id")) == qid:
                        qrows[j] = {"id": str(qid), "question": old_row.get("question", "")}
                        break
                _write_jsonl(QUESTIONS_PATH, qrows)
            return old_row
    return None


# ── Фиксация версии ───────────────────────────────────────────────────────────

def fix_version() -> str | None:
    """Архивирует _old в versions/<ts>.jsonl и кладёт рядом diff. Возвращает ts."""
    if not is_draft():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(_versions_dir(), f"answers_gold_{ts}.jsonl")
    diff_path = os.path.join(_versions_dir(), f"answers_gold_{ts}.diff.md")

    old_rows = _read_jsonl(_old_path())
    new_rows = _read_jsonl(_active_path())
    diff_md = _make_diff_md(old_rows, new_rows, archived_ts=ts)
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(diff_md)

    shutil.move(_old_path(), archive)
    return ts


def _make_diff_md(old_rows: list[dict], new_rows: list[dict], archived_ts: str) -> str:
    old_map = {str(r.get("id")): r for r in old_rows}
    new_map = {str(r.get("id")): r for r in new_rows}

    changed = []
    for qid, n in new_map.items():
        if qid not in old_map:
            changed.append((qid, "added", None, n))
            continue
        o = old_map[qid]
        if (
            o.get("question") != n.get("question")
            or o.get("answer") != n.get("answer")
            or o.get("context") != n.get("context")
        ):
            changed.append((qid, "changed", o, n))
    for qid, o in old_map.items():
        if qid not in new_map:
            changed.append((qid, "removed", o, None))

    out = [
        f"# Gold diff — archived as version `{archived_ts}`\n\n",
        f"_Generated at {datetime.now(timezone.utc).isoformat()} UTC_\n\n",
        f"**Total qids in archived (old) version:** {len(old_map)}\n",
        f"**Total qids in new active version:** {len(new_map)}\n",
        f"**Changed qids:** {len(changed)}\n\n",
        "---\n\n",
    ]
    if not changed:
        out.append("_(No content changes — versions are identical.)_\n")
        return "".join(out)

    for qid, kind, o, n in changed:
        out.append(f"## qid `{qid}` — {kind}\n\n")
        if kind in ("changed", "removed") and o:
            if kind == "changed" and o.get("question") != n.get("question"):
                out.append(f"**OLD question:** {o.get('question','')}\n\n")
            elif kind == "removed":
                out.append(f"**OLD question:** {o.get('question','')}\n\n")
            if kind == "changed" and o.get("answer") != n.get("answer"):
                out.append("**OLD answer:**\n\n```\n")
                out.append(o.get("answer", "") or "")
                out.append("\n```\n\n")
            elif kind == "removed":
                out.append("**OLD answer:**\n\n```\n")
                out.append(o.get("answer", "") or "")
                out.append("\n```\n\n")
        if kind in ("changed", "added") and n:
            if kind == "changed" and (not o or o.get("question") != n.get("question")):
                out.append(f"**NEW question:** {n.get('question','')}\n\n")
            elif kind == "added":
                out.append(f"**NEW question:** {n.get('question','')}\n\n")
            if kind == "changed" and (not o or o.get("answer") != n.get("answer")):
                out.append("**NEW answer:**\n\n```\n")
                out.append(n.get("answer", "") or "")
                out.append("\n```\n\n")
            elif kind == "added":
                out.append("**NEW answer:**\n\n```\n")
                out.append(n.get("answer", "") or "")
                out.append("\n```\n\n")
        if kind == "changed" and o and n and o.get("context") != n.get("context"):
            out.append(f"**OLD context:** {o.get('context','')}\n\n")
            out.append(f"**NEW context:** {n.get('context','')}\n\n")
        elif kind == "added" and n and n.get("context"):
            out.append(f"**NEW context:** {n.get('context','')}\n\n")
        elif kind == "removed" and o and o.get("context"):
            out.append(f"**OLD context:** {o.get('context','')}\n\n")
        out.append("---\n\n")
    return "".join(out)


def read_diff(version_ts: str) -> str | None:
    p = os.path.join(_versions_dir(), f"answers_gold_{version_ts}.diff.md")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()
