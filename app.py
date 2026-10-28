# app.py — MO Rag Evaluation — LLM-as-a-Judge с правкой gold и версионностью
import os
import json
import html
import time
import threading
import pandas as pd
import gradio as gr
from pathlib import Path

from markdown_it import MarkdownIt

from src.submission.check_validity import check_submission
from src.submission.submit import evaluate_submission, get_progress, SECONDARY_MODEL_EFFECTIVE
from src.envs import load_jsonl, QUESTIONS_PATH, EVAL_CONCURRENCY, EDIT_PASSWORDS, EVAL_MODEL

# Короткие имена судей для UI
JUDGE1 = EVAL_MODEL.split("/")[-1]
JUDGE2 = SECONDARY_MODEL_EFFECTIVE.split("/")[-1]
from src import gold_versioning as gv

# gfm-like — поддержка таблиц + strikethrough. html: False — raw HTML в ответах режется.
# linkify — голые URL → ссылки. breaks — \n становится <br>.
_md = MarkdownIt("gfm-like", {"html": False, "linkify": True, "breaks": True})


def _md_to_html(text: str) -> str:
    if not text:
        return ""
    return _md.render(str(text))


# CSS для карточек — инжектится один раз через gr.HTML. Поддерживает обе темы:
# Gradio навешивает класс `dark` на верхнем элементе при тёмной теме.
_CARD_CSS = """
<style>
.det-box { border-radius:8px; padding:12px; margin-bottom:0; font-size:13px; color:#1a1a1a; }
.det-box b { color:#1a1a1a; }
.det-box.det-correct  { background:#eaffea; }
.det-box.det-wrong    { background:#ffeaea; }
.det-box.det-tocheck  { background:#fffbe6; }
.det-box.det-noanswer { background:#f3f3f3; }

.det-box .qid-badge { background:#222; color:#fff; padding:2px 6px; border-radius:4px; font-size:11px; }
.det-box a { color:#0066cc; }
.det-box ul, .det-box ol { margin:4px 0 4px 22px; padding:0; }
.det-box li { margin:2px 0; }
.det-box p  { margin:4px 0; }
.det-box code { background:rgba(0,0,0,0.06); padding:1px 4px; border-radius:3px; font-size:90%; }
.det-box pre  { background:#f6f6f6; padding:8px; border-radius:4px; overflow-x:auto; margin:4px 0; }
.det-box pre code { background:transparent; padding:0; }
.det-box h1, .det-box h2, .det-box h3, .det-box h4 { margin:6px 0 4px; font-size:14px; }
.det-box table { border-collapse:collapse; margin:6px 0; font-size:12px; }
.det-box th, .det-box td { border:1px solid #bbb; padding:4px 8px; text-align:left; vertical-align:top; }
.det-box th { background:rgba(0,0,0,0.05); font-weight:600; }
.det-box hr.gp-sep { border:none; border-top:1px solid rgba(128,128,128,0.4); margin:10px 0; }
.det-box .hbtn {
  background: rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.18); border-radius:4px;
  padding: 1px 6px; margin: 0 2px; cursor: pointer; font-size: 13px;
}
.det-box .hbtn:hover { background: rgba(0,0,0,0.12); }
.det-box .hbtn.hbtn-active { background: rgba(0,0,0,0.25); border-color: rgba(0,0,0,0.5); }
.dark .det-box .hbtn { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.18); color: #e8e8e8; }
.dark .det-box .hbtn:hover { background: rgba(255,255,255,0.15); }
.dark .det-box .hbtn.hbtn-active { background: rgba(255,255,255,0.28); border-color: rgba(255,255,255,0.55); }

/* Edit-panel — изначально скрыта, JS показывает через inline display */
.gr-group:has(.edit-panel-anchor) { display: none; }
.gradio-container :has(> .edit-panel-anchor) { display: none; }

/* Human-bridge — невидимая служебная группа для JS-bridge */
.gr-group:has(.human-bridge-anchor) { display: none; }

/* ── Dark-theme overrides ── */
.dark .det-box, .dark .det-box b { color:#e8e8e8; }
.dark .det-box a { color:#7ab7ff; }
.dark .det-box.det-correct  { background:#1f3a26; }
.dark .det-box.det-wrong    { background:#3a1f1f; }
.dark .det-box.det-tocheck  { background:#3a341a; }
.dark .det-box.det-noanswer { background:#2a2a2a; }
.dark .det-box .qid-badge { background:#bbb; color:#1a1a1a; }
.dark .det-box code { background:rgba(255,255,255,0.08); }
.dark .det-box pre  { background:#15161a; }
.dark .det-box th, .dark .det-box td { border-color:#555; }
.dark .det-box th { background:rgba(255,255,255,0.06); }
</style>
"""

_CSS_CLASS_FOR = {
    "correct":   "det-correct",
    "wrong":     "det-wrong",
    "to_check":  "det-tocheck",
    "noanswer":  "det-noanswer",
}


def _sync_questions_from_active_gold() -> int:
    """Перезаписывает QUESTIONS_PATH вопросами из текущего active gold.
    Возвращает количество строк. Вызывается при build_ui — гарантия что
    Download questions всегда отдаёт последнюю версию gold."""
    try:
        rows = gv._read_jsonl(gv._active_path())
    except FileNotFoundError:
        return 0
    out = [{"id": str(r.get("id", "")), "question": r.get("question", "") or ""} for r in rows]
    try:
        os.makedirs(os.path.dirname(QUESTIONS_PATH), exist_ok=True)
        with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[questions-sync] failed: {e}")
    return len(out)

LEADERBOARD_PATH = "leaderboard.csv"
DETAILS_PATH = "eval_details.jsonl"
HUMAN_OVERRIDES_PATH = os.getenv("HUMAN_OVERRIDES_PATH", "data/human_overrides.json")


# ── Human overrides ──────────────────────────────────────────────────────────
# Структура: { submission_label: { qid: "correct"|"wrong"|"to_check" } }

_overrides_lock = threading.Lock()


def _load_overrides() -> dict:
    if not os.path.exists(HUMAN_OVERRIDES_PATH):
        return {}
    try:
        with open(HUMAN_OVERRIDES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_overrides(d: dict) -> None:
    os.makedirs(os.path.dirname(HUMAN_OVERRIDES_PATH) or ".", exist_ok=True)
    tmp = HUMAN_OVERRIDES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, HUMAN_OVERRIDES_PATH)


def _get_override(label: str, qid: str) -> str | None:
    return _load_overrides().get(label, {}).get(str(qid))


def _toggle_override(label: str, qid: str, status: str) -> str | None:
    """Если уже такой же — снимаем (None). Иначе ставим. Возвращает итог."""
    with _overrides_lock:
        d = _load_overrides()
        sub = d.setdefault(label, {})
        cur = sub.get(str(qid))
        if cur == status:
            sub.pop(str(qid), None)
            new = None
        else:
            sub[str(qid)] = status
            new = status
        if not sub:
            d.pop(label, None)
        _save_overrides(d)
    return new


def _count_overrides(label: str) -> int:
    return len(_load_overrides().get(label, {}))

# Файлы для скачивания
QUESTIONS_DOWNLOAD_PATH = os.getenv("QUESTIONS_DOWNLOAD_PATH", QUESTIONS_PATH)
DOCUMENTS_ZIP_PATH = os.getenv("DOCUMENTS_ZIP_PATH", "data/documents.zip")

LB_COLUMNS = [
    "filename",
    "Wrong", "Correct", "ToCheck", "LinksOK",
    "accuracy", "n_submit", "n_valid", "total",
    "gold_version", "eval_time", "timestamp",
]
# В Dataframe рендерим ещё одну колонку #HumanChg (вычисляется on-the-fly).
LB_DISPLAY_COLUMNS = LB_COLUMNS + ["#HumanChg"]

# Фоновые задачи: filename -> статус
_bg_jobs: dict[str, str] = {}
_bg_lock = threading.Lock()


# ── Leaderboard ───────────────────────────────────────────────────────────────

def ensure_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        pd.DataFrame(columns=LB_COLUMNS).to_csv(LEADERBOARD_PATH, index=False)
        return
    df = pd.read_csv(LEADERBOARD_PATH)
    changed = False
    for col in LB_COLUMNS:
        if col not in df.columns:
            # Backfill старых строк меткой 'legacy' для gold_version
            df[col] = "legacy" if col == "gold_version" else ""
            changed = True
    extra = [c for c in df.columns if c not in LB_COLUMNS]
    if extra:
        df = df.drop(columns=extra)
        changed = True
    if changed:
        df.to_csv(LEADERBOARD_PATH, index=False)


def sort_leaderboard(df):
    return df.sort_values(
        by=["accuracy", "Correct"],
        ascending=[False, False],
    ).reset_index(drop=True)


def load_sorted_leaderboard():
    ensure_leaderboard()
    df = pd.read_csv(LEADERBOARD_PATH)
    if df.empty:
        return df
    df = sort_leaderboard(df).reset_index(drop=False)
    df["Place"] = df["index"] + 1
    # Динамическая колонка: количество ручных override-ов для этого сабмита.
    def _human_chg_for_row(r):
        label = f"{r.get('timestamp','')} — {r.get('filename','')} [{r.get('gold_version','legacy')}]"
        return _count_overrides(label)
    df["#HumanChg"] = df.apply(_human_chg_for_row, axis=1)
    return df[["Place"] + LB_DISPLAY_COLUMNS]


# ── Eval details ──────────────────────────────────────────────────────────────

def load_all_detail_records() -> list[dict]:
    if not os.path.exists(DETAILS_PATH):
        return []
    records = []
    with open(DETAILS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def save_detail_record(filename: str, timestamp: str, gold_version: str, details: list) -> None:
    record = {
        "filename": filename,
        "timestamp": timestamp,
        "gold_version": gold_version,
        "details": details,
    }
    with open(DETAILS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_submissions() -> list[str]:
    records = load_all_detail_records()
    return [
        f"{r['timestamp']} — {r['filename']} [{r.get('gold_version','legacy')}]"
        for r in reversed(records)
    ]


def _render_sources_html(sources: list) -> str:
    """Принимает список: строки (URL), либо dict {name?, url} / {file, page}."""
    if not sources:
        return ""
    items = []
    for s in sources:
        if isinstance(s, str):
            url = html.escape(s)
            items.append(f'<li><a href="{url}" target="_blank">{url}</a></li>')
        elif isinstance(s, dict):
            if s.get("url"):
                u = html.escape(str(s["url"]))
                name = s.get("name")
                label = html.escape(str(name)) if name else u
                items.append(f'<li><a href="{u}" target="_blank">{label}</a></li>')
            elif s.get("file"):
                file = html.escape(str(s["file"]))
                page = s.get("page")
                page_part = f", стр. {html.escape(str(page))}" if page is not None else ""
                items.append(f"<li>📄 {file}{page_part}</li>")
            else:
                items.append(f"<li>{html.escape(json.dumps(s, ensure_ascii=False))}</li>")
        else:
            items.append(f"<li>{html.escape(str(s))}</li>")
    return "<ul style='margin:4px 0 0 18px;padding:0;'>" + "".join(items) + "</ul>"


def _render_gold_context_html(gold_context: str) -> str:
    """gold.context для МОРАГ имеет формат "Раздел: ...\\nИсточник: <url>"."""
    if not gold_context:
        return ""
    # Извлекаем все http(s)-ссылки
    import re
    urls = re.findall(r"https?://\S+", gold_context)
    parts = []
    for line in gold_context.splitlines():
        if line.strip():
            line_html = html.escape(line)
            # Прокликнуть URL'ы
            for u in re.findall(r"https?://\S+", line):
                u_esc = html.escape(u)
                line_html = line_html.replace(
                    u_esc, f'<a href="{u_esc}" target="_blank">{u_esc}</a>'
                )
            parts.append(line_html)
    return "<br>".join(parts)


# ── Downloads ─────────────────────────────────────────────────────────────────

def download_questions():
    if os.path.exists(QUESTIONS_DOWNLOAD_PATH):
        return QUESTIONS_DOWNLOAD_PATH
    return None


def download_documents():
    if os.path.exists(DOCUMENTS_ZIP_PATH):
        return DOCUMENTS_ZIP_PATH
    return None


# ── Background eval ───────────────────────────────────────────────────────────

def _run_eval_bg(file_path: str, filename: str, use_secondary: bool, gold_version: str):
    t_start = time.time()
    try:
        result = evaluate_submission(
            file_path,
            use_secondary=use_secondary,
            gold_version=gold_version,
        )
    except Exception as e:
        with _bg_lock:
            _bg_jobs[filename] = f"❌ Evaluation failed: {e}"
        return

    eval_time = round(time.time() - t_start, 1)
    n_submit = result["n_submit"]
    n_valid = result.get("n_valid", n_submit)
    total = result["total"]
    correct = result["ones"]
    wrong = result["zeros"]
    to_check = result.get("to_check", 0)
    links_ok = result.get("links_ok", 0)
    accuracy = round(correct / max(n_valid, 1), 4)
    details = result.get("details", [])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    try:
        save_detail_record(filename, timestamp, gold_version, details)
    except Exception as e:
        print(f"Warning: could not save details: {e}")

    row = {
        "timestamp": timestamp, "filename": filename,
        "Wrong": wrong, "Correct": correct, "ToCheck": to_check,
        "LinksOK": f"{links_ok}/{n_submit}",
        "accuracy": accuracy,
        "n_submit": n_submit, "n_valid": n_valid, "total": total,
        "gold_version": gold_version,
        "eval_time": f"{eval_time}s",
    }
    df = pd.read_csv(LEADERBOARD_PATH)
    df.loc[len(df)] = row
    df.to_csv(LEADERBOARD_PATH, index=False)

    status = (
        f"✅ **Done!** `{filename}` — gold `{gold_version}` — "
        f"Answered: {n_submit}/{total} | Correct: {correct} | Wrong: {wrong} | "
        f"ToCheck: {to_check} | LinksOK: {links_ok}/{n_submit} | "
        f"Accuracy: {accuracy:.1%} | Time: {eval_time}s"
    )
    with _bg_lock:
        _bg_jobs[filename] = status


# ── Submit ────────────────────────────────────────────────────────────────────

def submit_file(file_obj, use_secondary: bool, gold_version: str):
    ensure_leaderboard()
    no_progress = gr.update(visible=False)

    if file_obj is None:
        return "❌ Please upload a JSONL file", load_sorted_leaderboard(), gr.update(choices=list_submissions()), no_progress

    file_path = file_obj.name
    filename = Path(file_path).name

    ok, msg = check_submission(file_path, QUESTIONS_PATH)
    if not ok:
        return f"❌ Invalid submission: {msg}", load_sorted_leaderboard(), gr.update(choices=list_submissions()), no_progress

    with _bg_lock:
        _bg_jobs[filename] = "⏳ Evaluating in background..."

    t = threading.Thread(
        target=_run_eval_bg,
        args=(file_path, filename, use_secondary, gold_version or "active"),
        daemon=True,
    )
    t.start()

    mode = f"{JUDGE1} + {JUDGE2} (dual)" if use_secondary else f"{JUDGE1} only"
    msg = f"⏳ Evaluating **{filename}** ({mode}) on gold `{gold_version}`... watch the progress bar below."
    return msg, load_sorted_leaderboard(), gr.update(choices=list_submissions()), gr.update(visible=True, value="⏳ Progress: 0 / ?")


def poll_progress() -> gr.update:
    """Вызывается таймером — обновляет строку прогресса."""
    from src.submission.submit import _progress, _progress_lock
    with _progress_lock:
        if not _progress:
            return gr.update(visible=False)
        filename, p = list(_progress.items())[-1]
        done = p.get("done", 0)
        total = p.get("total", 0)
        cached = p.get("cached", 0)
        pn = p.get("pn", 0)
        pt = p.get("pt", 0.0)
        sn = p.get("sn", 0)
        st = p.get("st", 0.0)
    if total == 0:
        return gr.update(visible=False)
    if done >= total:
        with _bg_lock:
            status = _bg_jobs.get(filename, "")
        if status.startswith("✅"):
            return gr.update(visible=True, value=status)
        return gr.update(visible=False)
    pct = int(done / total * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    avg_p = f"{pt/pn:.1f}s" if pn else "—"
    avg_s = f"{st/sn:.1f}s" if sn else "—"
    remaining = total - done
    if sn > 0:
        eta_sec = (st / sn) * remaining / max(EVAL_CONCURRENCY, 1)
    elif pn > 0:
        eta_sec = (pt / pn) * remaining / max(EVAL_CONCURRENCY, 1)
    else:
        eta_sec = 0
    if eta_sec >= 60:
        eta_str = f"{int(eta_sec//60)}m {int(eta_sec%60)}s"
    else:
        eta_str = f"{int(eta_sec)}s"
    return gr.update(
        visible=True,
        value=f"⏳ `{bar}` {done}/{total} ({pct}%) — {remaining} left ~{eta_str} | {JUDGE1}: {avg_p}/q | {JUDGE2}: {avg_s}/q | cache: {cached}"
    )


# ── Edit gold helpers ─────────────────────────────────────────────────────────

def _draft_status_md() -> str:
    if gv.is_draft():
        return "🔓 **Draft** — есть не зафиксированные правки. Жми «💾 Save New Gold Set» когда готов."
    return "🔒 **Fixed** — текущий active gold зафиксирован."


def _detail_record_by_label(label: str) -> dict | None:
    if not label:
        return None
    for r in load_all_detail_records()[::-1]:
        if f"{r['timestamp']} — {r['filename']} [{r.get('gold_version','legacy')}]" == label:
            return r
    return None


def _current_gold_for_qid(qid: str) -> tuple[str, str, str]:
    row = gv.get_row(qid)
    if not row:
        return "", "", ""
    return (
        row.get("question", "") or "",
        row.get("answer", "") or "",
        row.get("context", "") or "",
    )


def _current_gold_context(qid: str) -> str:
    row = gv.get_row(qid)
    return (row.get("context", "") if row else "") or ""


def _render_qid_card_html(d: dict, css_class: str) -> str:
    """Read-only представление одной qid-строки (без кнопок)."""
    qid = html.escape(str(d.get("id", "")))
    question = html.escape(str(d.get("question", "")))
    # Gold/Pred рендерим как markdown — поддерживается **bold**, списки, заголовки, код.
    gold = _md_to_html(d.get("gold", ""))
    pred = _md_to_html(d.get("pred", ""))
    sp = d.get("score_primary", "?")
    ss = d.get("score_secondary", "?")
    verdict_line = f"<b>Судья 1:</b> {sp} &nbsp;|&nbsp; <b>Судья 2:</b> {ss}"
    if d.get("sources"):
        lo = d.get("links_ok")
        if lo is not None:
            verdict_line += f' &nbsp;|&nbsp; <b>🔗 Link:</b> {"✓" if lo else "✗"}'
    h = d.get("human_override")
    if h:
        label_map = {"correct": "🟢 Correct", "wrong": "🔴 Wrong", "to_check": "🟡 ToCheck"}
        verdict_line += f' &nbsp;|&nbsp; <b>Human:</b> {label_map.get(h, h)}'

    qid_attr = html.escape(str(d.get("id", "")))
    btn_active_correct = "hbtn-active" if h == "correct" else ""
    btn_active_wrong = "hbtn-active" if h == "wrong" else ""
    btn_active_tocheck = "hbtn-active" if h == "to_check" else ""
    change_to_row = (
        '<div class="human-row" style="margin-top:6px;font-size:12px;">'
        '<b>Change to:</b> '
        f'<button class="hbtn {btn_active_correct}" data-qid="{qid_attr}" data-s="correct" title="Correct">🟢</button> '
        f'<button class="hbtn {btn_active_wrong}" data-qid="{qid_attr}" data-s="wrong" title="Wrong">🔴</button> '
        f'<button class="hbtn {btn_active_tocheck}" data-qid="{qid_attr}" data-s="to_check" title="ToCheck">🟡</button>'
        '</div>'
    )
    sources_html = _render_sources_html(d.get("sources") or [])
    # gold.context старых сабмитов мог быть не сохранён в detail-record —
    # подтягиваем актуальный из active gold по qid.
    gold_ctx_raw = d.get("gold_context") or _current_gold_context(str(d.get("id", "")))
    gold_ctx_html = _render_gold_context_html(gold_ctx_raw or "")
    sources_block = ""
    if sources_html:
        sources_block += f"<br><b>📎 Источник модели:</b>{sources_html}"
    if gold_ctx_html:
        sources_block += f"<br><b>📌 Источник:</b><br>{gold_ctx_html}"
    if not sources_html and not gold_ctx_html:
        sources_block = "<br><i>📭 Источников нет (ни в gold, ни в pred)</i>"
    return f"""
<div class="det-box {css_class}">
  <span class="qid-badge">qid {qid}</span> &nbsp; {question}<br>
  <b>Gold:</b> <div>{gold}</div>
  <hr class="gp-sep">
  <b>Pred:</b> <div>{pred}</div>
  {sources_block}
  <br>{verdict_line}
  {change_to_row}
</div>"""


def _class_for_detail(d: dict) -> str:
    # human override (если есть) переопределяет judge-status для bg-класса
    h = d.get("human_override")
    if h == "correct":  return _CSS_CLASS_FOR["correct"]
    if h == "wrong":    return _CSS_CLASS_FOR["wrong"]
    if h == "to_check": return _CSS_CLASS_FOR["to_check"]
    if d.get("to_check"):
        return _CSS_CLASS_FOR["to_check"]
    return _CSS_CLASS_FOR["correct"] if d.get("score") == 1 else _CSS_CLASS_FOR["wrong"]


def _render_card_no_answer(qid: str, question: str, gold: str, context: str) -> str:
    qid_e = html.escape(str(qid))
    question_e = html.escape(question)
    gold_html = _md_to_html(gold)
    gold_ctx_html = _render_gold_context_html(context or "")
    if gold_ctx_html:
        src_block = f"<br><b>📌 Источник:</b><br>{gold_ctx_html}"
    else:
        src_block = "<br><i>📭 Источников нет (ни в gold, ни в pred)</i>"
    return f"""
<div class="det-box det-noanswer">
  <span class="qid-badge">qid {qid_e}</span> &nbsp; {question_e}<br>
  <b>Gold:</b> <div>{gold_html}</div>
  {src_block}
  <br><i>(нет ответа модели в этом сабмите)</i>
</div>"""


def _render_counters_md(record: dict | None) -> str:
    if not record:
        return ""
    counts = {"correct": 0, "wrong": 0, "to_check": 0, "answered": 0}
    for d in record.get("details", []):
        counts["answered"] += 1
        if d.get("to_check"):
            counts["to_check"] += 1
        elif d.get("score") == 1:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return (
        f"**Сабмит:** answered {counts['answered']} | "
        f"🟢 {counts['correct']} | 🔴 {counts['wrong']} | 🟡 {counts['to_check']}"
    )


def _label_for_record(record: dict | None) -> str:
    if not record:
        return ""
    return f"{record.get('timestamp','')} — {record.get('filename','')} [{record.get('gold_version','legacy')}]"


def _attach_override(d: dict, label: str) -> dict:
    if not label:
        return d
    h = _get_override(label, str(d.get("id", "")))
    if h:
        d = {**d, "human_override": h}
    return d


def _initial_card_html(qid: str, gold_row: dict, record: dict | None) -> str:
    label = _label_for_record(record)
    if record:
        for d in record.get("details", []):
            if str(d.get("id", "")) == qid:
                d2 = _attach_override(d, label)
                return _render_qid_card_html(d2, _class_for_detail(d2))
    return _render_card_no_answer(
        qid,
        gold_row.get("question", "") or "",
        gold_row.get("answer", "") or "",
        gold_row.get("context", "") or "",
    )


def _open_edit(qid: str):
    """Кнопка ✏️ Edit в карточке: загружает qid в общую edit-панель.
    Видимостью панели управляет JS (inline display), здесь только данные."""
    cq, ca, cctx = _current_gold_for_qid(qid)
    return (
        qid,                              # editing_qid_state
        f"### ✏️ Редактирование `qid {qid}`",
        cq, ca, cctx,
        "",                                # edit_msg — очистить
    )


def submit_human_overrides(current_label: str):
    """Создаёт новый сабмит-record на основе текущего + human overrides.
    Применяет overrides к каждому detail (score/to_check), пересчитывает счётчики,
    добавляет строку в LB с filename + '_HumChg' suffix."""
    if not current_label:
        return "❌ Не выбран сабмит.", gr.update(), gr.update()
    record = _detail_record_by_label(current_label)
    if not record:
        return f"❌ Не найден сабмит `{current_label}`.", gr.update(), gr.update()
    overrides = _load_overrides().get(current_label, {})
    if not overrides:
        return "ℹ️ Нет ручных правок для этого сабмита — нечего сабмитить.", gr.update(), gr.update()

    # Применяем overrides
    new_details = []
    for d in record.get("details", []):
        new_d = {**d}
        qid = str(d.get("id", ""))
        h = overrides.get(qid)
        if h == "correct":
            new_d["score"] = 1
            new_d["to_check"] = False
        elif h == "wrong":
            new_d["score"] = 0
            new_d["to_check"] = False
        elif h == "to_check":
            new_d["to_check"] = True
        if h:
            new_d["human_override"] = h
        new_details.append(new_d)

    # Пересчёт счётчиков (тех же что в _run_eval_bg)
    answered = [d for d in new_details if (d.get("pred") or "").strip()]
    n_submit = len(answered)
    to_check = sum(1 for d in new_details if d.get("to_check"))
    correct = sum(1 for d in new_details if d.get("score") == 1 and not d.get("to_check"))
    wrong = sum(1 for d in new_details if d.get("score") == 0 and not d.get("to_check"))
    n_valid = max(n_submit - to_check, 0)
    total = record.get("total") or len(new_details)
    accuracy = round(correct / max(n_valid, 1), 4)
    links_ok = sum(int(d.get("links_ok") or 0) for d in new_details)

    # Имя файла — добавляем _HumChg перед расширением (если уже есть — наращиваем счётчик _HumChg2 и т.д.)
    orig_fn = record.get("filename", "submission.jsonl")
    base, _, ext = orig_fn.rpartition(".")
    if not base:
        base = orig_fn
        ext = ""
    if "_HumChg" in base:
        # уже есть суффикс — добавляем порядковый
        import re as _re
        m = _re.match(r"^(.*_HumChg)(\d*)$", base)
        if m:
            n = int(m.group(2) or "1") + 1
            base = f"{m.group(1)}{n}"
    else:
        base = base + "_HumChg"
    new_fn = f"{base}.{ext}" if ext else base

    new_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    gold_version = record.get("gold_version", "legacy")

    save_detail_record(new_fn, new_ts, gold_version, new_details)

    row = {
        "timestamp": new_ts, "filename": new_fn,
        "Wrong": wrong, "Correct": correct, "ToCheck": to_check,
        "LinksOK": f"{links_ok}/{n_submit}",
        "accuracy": accuracy,
        "n_submit": n_submit, "n_valid": n_valid, "total": total,
        "gold_version": gold_version,
        "eval_time": "0s (human)",
    }
    df = pd.read_csv(LEADERBOARD_PATH)
    df.loc[len(df)] = row
    df.to_csv(LEADERBOARD_PATH, index=False)

    subs = list_submissions()
    msg = (
        f"✅ Сабмит `{new_fn}` создан на основе `{orig_fn}` + {len(overrides)} ручных правок. "
        f"Correct: {correct} | Wrong: {wrong} | ToCheck: {to_check} | Accuracy: {accuracy:.1%}"
    )
    return msg, load_sorted_leaderboard(), gr.update(choices=subs)


def fix_action():
    """«💾 Save New Gold Set» — архивирует draft + показывает diff."""
    if not gv.is_draft():
        return (
            "ℹ️ Нет draft — фиксировать нечего.",
            _draft_status_md(),
            gr.update(),
            gr.update(),
            render_versions_md(),
            "_(нет diff — нечего показать)_",
        )
    ts = gv.fix_version()
    diff_md = gv.read_diff(ts) or "_(diff пуст)_"
    return (
        f"✅ Версия зафиксирована как `{ts}`. Diff показан внизу.",
        _draft_status_md(),
        gr.update(choices=gv.submit_dropdown_choices()),
        gr.update(choices=[v["ts"] for v in gv.list_versions()], value=ts),
        render_versions_md(),
        diff_md,
    )


# ── Versions tab ──────────────────────────────────────────────────────────────

def render_versions_md() -> str:
    versions = gv.list_versions()
    lines = [f"**Состояние:** {_draft_status_md()}\n"]
    if not versions:
        lines.append("\n_Архивных версий пока нет. После первой правки и «Fix version» здесь появится список._")
        return "\n".join(lines)
    lines.append("\n| Version (UTC) | qids | Diff |")
    lines.append("|---|---|---|")
    for v in versions:
        diff_link = "✅" if v["diff_path"] else "—"
        lines.append(f"| `{v['ts']}` | {v['rows']} | {diff_link} |")
    return "\n".join(lines)


def render_diff(version_ts: str) -> str:
    if not version_ts:
        return "_(Выбери версию выше.)_"
    md = gv.read_diff(version_ts)
    if md is None:
        return f"❌ diff для `{version_ts}` не найден."
    return md


# ── UI ────────────────────────────────────────────────────────────────────────

def _login(pwd: str):
    if pwd in EDIT_PASSWORDS:
        return (
            gr.update(visible=False),  # gate
            gr.update(visible=True),   # main
            "",                         # очистить пароль
            gr.update(visible=False),   # err
        )
    return (
        gr.update(),
        gr.update(),
        "",
        gr.update(visible=True, value="❌ Неверный пароль"),
    )


def build_ui():
    ensure_leaderboard()
    _sync_questions_from_active_gold()

    with gr.Blocks(title="MO Rag Evaluation") as demo:

        # ── 0. Pwd-gate ───────────────────────────────────────────────────────
        with gr.Column(visible=True) as gate:
            with gr.Row():
                gr.Markdown("&nbsp;")
            with gr.Row():
                gr.Markdown("&nbsp;")
            gr.Markdown("# 🔒 MO Rag Evaluation", elem_id="gate_title")
            gr.Markdown("Введите пароль для входа.")
            pwd_in = gr.Textbox(label="Пароль", type="password", autofocus=True)
            login_btn = gr.Button("Войти", variant="primary")
            login_err = gr.Markdown(visible=False)

        # ── 1. Основная страница (скрыта до авторизации) ──────────────────────
        with gr.Column(visible=False) as main:
            gr.HTML(_CARD_CSS)
            gr.Markdown(
                "# 🏁 MO Rag Evaluation — LLM-as-a-Judge\n"
                "Загрузи ответы своей RAG-системы в JSONL. "
                f"Каждый ответ судит **{JUDGE1}** (опционально + **{JUDGE2}**) против gold."
            )

            # ── Leaderboard ─────────────────────────────────────────────────
            gr.Markdown("## 📊 Leaderboard")
            out_df = gr.Dataframe(value=load_sorted_leaderboard(), interactive=False, wrap=True, label="")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

            gr.Markdown("---")

            # ── Downloads ───────────────────────────────────────────────────
            gr.Markdown("## 📥 Downloads")
            gr.DownloadButton(
                label="📋 Download questions (JSONL) — текущий active gold",
                value=QUESTIONS_DOWNLOAD_PATH if os.path.exists(QUESTIONS_DOWNLOAD_PATH) else None,
                variant="secondary",
            )

            gr.Markdown("---")

            # ── Submit ──────────────────────────────────────────────────────
            gr.Markdown(
                "## 📤 Submit your answers\n\n"
                "**Format** — one JSON per line:\n"
                "```json\n"
                "{\"id\": \"0\", \"answer\": \"Your answer here\", \"sources\": [\"https://...\", {\"file\":\"doc.pdf\",\"page\":12}]}\n"
                "```\n"
                "Поле `sources` опциональное — массив URL'ов или объектов `{file, page}`. "
                "Если есть, отобразится в деталях рядом с ответом.\n\n"
                "⚠️ Evaluation runs in the background. Press **🔄 Refresh** when done."
            )
            with gr.Row():
                gold_version_in = gr.Dropdown(
                    choices=gv.submit_dropdown_choices(),
                    value="active",
                    label="Gold version (против какой версии судить)",
                    interactive=True,
                )
                use_secondary_in = gr.Checkbox(
                    value=False,
                    label=f"Secondary judge ({JUDGE2}) — dual-judge с ToCheck",
                )
            file_in = gr.File(label="Upload JSONL (answers)", file_types=[".jsonl"])
            submit_btn = gr.Button("Submit", variant="primary")
            out_msg = gr.Markdown()
            progress_bar = gr.Markdown(visible=False)

            # Timer для poll_progress объявляем здесь, до Details-секции.
            _progress_timer = gr.Timer(value=3)
            _progress_timer.tick(fn=poll_progress, inputs=[], outputs=[progress_bar])

            gr.Markdown("---")

            # ── Dataset info ────────────────────────────────────────────────
            gr.Markdown(
                "## 📋 Dataset info\n"
                f"- **Primary judge:** `{JUDGE1}` (OpenRouter)\n"
                f"- **Secondary judge:** `{JUDGE2}` (OpenRouter, опционально)\n"
                "- **Scoring:** оба согласны → ✅ Correct/❌ Wrong; разошлись → 🟡 ToCheck (исключается из accuracy)\n"
                "- **Accuracy** = Correct / (n_submit − ToCheck)\n"
                "- Gold answers: stored privately on server\n"
            )
            gr.Markdown("---")

            # ── Details + Edit (статика, без перерисовки) ────────────────
            gr.Markdown("## 🔍 Evaluation details + ✏️ Edit gold")
            details_dropdown = gr.Dropdown(
                choices=list_submissions(),
                value=list_submissions()[0] if list_submissions() else None,
                label="Select submission",
                interactive=True,
            )
            with gr.Row():
                draft_status = gr.Markdown(value=_draft_status_md())
                fix_btn = gr.Button("💾 Save New Gold Set", variant="primary")
            fix_msg = gr.Markdown()

            # Все qid'ы из gold создаются как статические компоненты —
            # никакой перерисовки на ✏️/Save, Accordion раскрывается на фронте.
            try:
                _gold_rows = gv._read_jsonl(gv._active_path())
            except FileNotFoundError:
                _gold_rows = []
            _last_record = (load_all_detail_records() or [None])[-1]

            counters_md = gr.Markdown(value=_render_counters_md(_last_record))

            with gr.Row():
                submit_human_btn = gr.Button("📨 Submit Human Overrides", variant="secondary", size="sm")
            submit_human_msg = gr.Markdown()

            # ── Скрытый JS-bridge для Human-override кнопок в HTML карточках ─
            # Группа visible=True (чтобы anchor был в DOM), скрыта через CSS :has().
            with gr.Group() as _human_bridge_group:
                gr.HTML('<span class="human-bridge-anchor" style="display:none"></span>')
                human_payload = gr.Textbox(value="", interactive=True)
                human_apply_btn = gr.Button("apply-human")

            # ── Одна общая edit-panel, скрыта пока не нажата ✏️ Edit ─────
            editing_qid_state = gr.State("")
            # Панель ВСЕГДА visible=True (в DOM), но скрыта через CSS :has().
            # JS показывает её inline-стилем при клике ✏️ Edit и переносит к карточке.
            with gr.Group() as edit_panel:
                gr.HTML('<span class="edit-panel-anchor" style="display:none"></span>')
                editing_qid_label = gr.Markdown("")
                edit_q = gr.Textbox(label="Question (active gold)", lines=2, interactive=True)
                edit_a = gr.Textbox(label="Gold answer (active gold)", lines=8, interactive=True)
                edit_ctx = gr.Textbox(
                    label="Источник (gold context)", lines=2, interactive=True,
                    placeholder="Раздел: ...\nИсточник: https://...",
                )
                with gr.Row():
                    save_b = gr.Button("💾 Save", variant="primary", size="sm")
                    reload_b = gr.Button("↩ Reload from active", size="sm")
                    close_b = gr.Button("✖ Close", size="sm")
                edit_msg = gr.Markdown()

            _card_html_components: list[gr.HTML] = []
            _card_qids: list[str] = []

            # JS-функция: переносит общую edit-панель в DOM сразу после
            # кликнутой карточки → пользователь правит inline, без скролла наверх.
            # Gradio запускает js ПЕРЕД fn и передаёт его return как вход для fn.
            _move_panel_js = """
(qid) => {
  const panelAnchor = document.querySelector('.edit-panel-anchor');
  const panel = panelAnchor ? panelAnchor.closest('.gr-group, .group') : null;
  const cardAnchor = document.querySelector('.qcard-anchor[data-qid="' + qid + '"]');
  const card = cardAnchor ? cardAnchor.closest('.gr-group, .group') : null;
  if (panel) {
    if (card && card.nextSibling !== panel) {
      card.parentNode.insertBefore(panel, card.nextSibling);
    }
    panel.style.display = 'flex';
    panel.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
  return qid;
}
"""

            _hide_panel_js = """
() => {
  const a = document.querySelector('.edit-panel-anchor');
  const p = a ? a.closest('.gr-group, .group') : null;
  if (p) p.style.display = 'none';
  return '';
}
"""

            for _r in _gold_rows:
                _qid = str(_r.get("id", ""))
                with gr.Group():
                    gr.HTML(f'<span class="qcard-anchor" data-qid="{html.escape(_qid)}" style="display:none"></span>')
                    _card = gr.HTML(value=_initial_card_html(_qid, _r, _last_record))
                    _edit_btn = gr.Button("✏️ Edit", size="sm")
                    # Скрытый Textbox несёт qid в js + в fn.
                    _qid_marker = gr.Textbox(value=_qid, visible=False)
                    _edit_btn.click(
                        fn=_open_edit,
                        inputs=[_qid_marker],
                        outputs=[
                            editing_qid_state,
                            editing_qid_label,
                            edit_q, edit_a, edit_ctx,
                            edit_msg,
                        ],
                        js=_move_panel_js,
                    )
                _card_html_components.append(_card)
                _card_qids.append(_qid)

            gr.Markdown("---")

            # ── Versions ────────────────────────────────────────────────────
            gr.Markdown("## 📚 Gold versions")
            versions_md = gr.Markdown(value=render_versions_md())
            with gr.Row():
                diff_select = gr.Dropdown(
                    choices=[v["ts"] for v in gv.list_versions()],
                    label="View diff for archived version",
                    interactive=True,
                )
                diff_btn = gr.Button("📜 Show diff")
            diff_view = gr.Markdown()

        # ── Привязка событий ─────────────────────────────────────────────────

        # Auth
        login_btn.click(_login, [pwd_in], [gate, main, pwd_in, login_err])
        pwd_in.submit(_login, [pwd_in], [gate, main, pwd_in, login_err])

        # Refresh
        def do_refresh():
            ensure_leaderboard()
            subs = list_submissions()
            return (
                load_sorted_leaderboard(),
                gr.update(choices=subs, value=subs[0] if subs else None),
                gr.update(choices=gv.submit_dropdown_choices()),
                _draft_status_md(),
                render_versions_md(),
                gr.update(choices=[v["ts"] for v in gv.list_versions()]),
            )

        refresh_outputs = [
            out_df, details_dropdown,
            gold_version_in, draft_status, versions_md, diff_select,
        ]
        refresh_btn.click(fn=do_refresh, inputs=[], outputs=refresh_outputs)
        # demo.load: JS навешивает глобальный делегированный listener на клики
        # по .hbtn (human-override кнопкам внутри HTML карточек). При клике
        # пишет payload в скрытый Textbox и триггерит скрытый Button.
        demo.load(
            fn=do_refresh,
            inputs=[],
            outputs=refresh_outputs,
            js="""
() => {
  if (!window.__hbtnInstalled) {
    window.__hbtnInstalled = true;
    document.body.addEventListener('click', (e) => {
      const t = e.target.closest('.hbtn');
      if (!t) return;
      e.preventDefault();
      const a = document.querySelector('.human-bridge-anchor');
      if (!a) return;
      const grp = a.closest('.gr-group, .group');
      if (!grp) return;
      const input = grp.querySelector('textarea, input[type=text]');
      const btn   = grp.querySelector('button');
      if (!input || !btn) return;
      input.value = JSON.stringify({qid: t.dataset.qid, status: t.dataset.s, ts: Date.now()});
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
      setTimeout(() => btn.click(), 80);
    });
  }
}
""",
        )

        # При смене сабмита — batch-обновить все HTML-карточки + counters
        def on_dropdown_change(label):
            record = _detail_record_by_label(label)
            by_qid = {}
            if record:
                by_qid = {str(d.get("id", "")): d for d in record.get("details", [])}
            updates = []
            for qid in _card_qids:
                d = by_qid.get(qid)
                if d:
                    d2 = _attach_override(d, label)
                    upd_html = _render_qid_card_html(d2, _class_for_detail(d2))
                else:
                    row = next((r for r in _gold_rows if str(r.get("id", "")) == qid), {})
                    upd_html = _render_card_no_answer(
                        qid,
                        row.get("question", "") or "",
                        row.get("answer", "") or "",
                        row.get("context", "") or "",
                    )
                updates.append(upd_html)
            return [_render_counters_md(record)] + updates

        details_dropdown.change(
            fn=on_dropdown_change,
            inputs=[details_dropdown],
            outputs=[counters_md] + _card_html_components,
        )

        # Submit
        submit_btn.click(
            fn=submit_file,
            inputs=[file_in, use_secondary_in, gold_version_in],
            outputs=[out_msg, out_df, details_dropdown, progress_bar],
        )

        # Save New Gold Set
        fix_btn.click(
            fn=fix_action,
            inputs=[],
            outputs=[fix_msg, draft_status, gold_version_in, diff_select, versions_md, diff_view],
        )

        # Versions diff
        diff_btn.click(fn=render_diff, inputs=[diff_select], outputs=[diff_view])

        # ── Human override: триггерится скрытым button из JS ───────────────
        def apply_human_override(payload, current_label):
            no_updates = [gr.update(value=load_sorted_leaderboard()), gr.update(value="")] + [gr.update() for _ in _card_qids]
            if not payload or not current_label:
                return no_updates
            try:
                obj = json.loads(payload)
            except Exception:
                return no_updates
            qid = str(obj.get("qid", ""))
            status = obj.get("status", "")
            if qid not in set(_card_qids) or status not in ("correct", "wrong", "to_check"):
                return no_updates
            _toggle_override(current_label, qid, status)

            # Перерисовать только эту карточку
            record = _detail_record_by_label(current_label)
            by_qid = {str(d.get("id", "")): d for d in record.get("details", [])} if record else {}
            updates = []
            for cqid in _card_qids:
                if cqid != qid:
                    updates.append(gr.update())
                    continue
                d = by_qid.get(cqid)
                if d:
                    d2 = _attach_override(d, current_label)
                    updates.append(_render_qid_card_html(d2, _class_for_detail(d2)))
                else:
                    row = next((r for r in _gold_rows if str(r.get("id", "")) == cqid), {})
                    updates.append(_render_card_no_answer(
                        cqid, row.get("question","") or "", row.get("answer","") or "", row.get("context","") or ""
                    ))
            return [load_sorted_leaderboard(), ""] + updates

        human_apply_btn.click(
            fn=apply_human_override,
            inputs=[human_payload, details_dropdown],
            outputs=[out_df, human_payload] + _card_html_components,
        )

        # Submit Human Overrides — создать новый сабмит с применёнными правками
        submit_human_btn.click(
            fn=submit_human_overrides,
            inputs=[details_dropdown],
            outputs=[submit_human_msg, out_df, details_dropdown],
        )

        # ── Общая edit-панель: Save / Reload / Close ────────────────────────
        def save_open(qid, q, a, ctx, current_submission_label):
            if not qid:
                return ["⚠️ Нет открытого qid.", _draft_status_md()] + [gr.update() for _ in _card_qids]
            try:
                gv.apply_edit(qid, question=q, answer=a, context=ctx)
                msg = f"💾 Saved `{qid}` — кэш судьи инвалидируется автоматически."
            except Exception as e:
                return [f"❌ Save failed: {e}", _draft_status_md()] + [gr.update() for _ in _card_qids]

            # Обновляем только эту карточку. Берём текущий выбранный сабмит из dropdown,
            # чтобы карточка осталась в той же «сабмит-вселенной», что и соседние.
            record = _detail_record_by_label(current_submission_label)
            by_qid = {str(d.get("id", "")): d for d in record.get("details", [])} if record else {}
            updates = []
            for cqid in _card_qids:
                if cqid != qid:
                    updates.append(gr.update())
                    continue
                d = by_qid.get(cqid)
                cur_q, cur_a, cur_ctx = _current_gold_for_qid(cqid)
                if d:
                    d2 = {**d, "question": cur_q, "gold": cur_a, "gold_context": cur_ctx}
                    d2 = _attach_override(d2, current_submission_label)
                    updates.append(_render_qid_card_html(d2, _class_for_detail(d2)))
                else:
                    updates.append(_render_card_no_answer(cqid, cur_q, cur_a, cur_ctx))
            return [msg, _draft_status_md()] + updates

        save_b.click(
            fn=save_open,
            inputs=[editing_qid_state, edit_q, edit_a, edit_ctx, details_dropdown],
            outputs=[edit_msg, draft_status] + _card_html_components,
        )

        def reload_open(qid):
            if not qid:
                return "", "", "", "⚠️ Нет открытого qid."
            cq, ca, cctx = _current_gold_for_qid(qid)
            return cq, ca, cctx, "↩ Reloaded from active gold."

        reload_b.click(
            fn=reload_open,
            inputs=[editing_qid_state],
            outputs=[edit_q, edit_a, edit_ctx, edit_msg],
        )

        def close_open():
            return "", ""

        close_b.click(
            fn=close_open,
            inputs=[],
            outputs=[editing_qid_state, edit_msg],
            js=_hide_panel_js,
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7862)
