# MO Rag Evaluation — развёртывание

Проверено на Ubuntu 22.04+, Python 3.11/3.12. Приложение — один процесс Gradio на порту **7862**,
данные — файлы в `data/` (БД не нужна).

## 1. Что понадобится

- Сервер с Python 3.11+ (1 vCPU / 1 ГБ RAM достаточно — судья работает через API).
- Ключ **OpenRouter** (`OPENROUTER_API_KEY`) — им ходят оба судьи.
- Архив с данными `mo-rag-eval-data-*.zip` (передаётся отдельно, в git его нет):
  эталоны, их версии, вопросы, документы.
- Для HTTPS — домен и nginx + certbot (опционально).

## 2. Код и окружение

```bash
sudo mkdir -p /opt/mo-rag-eval && sudo chown $USER /opt/mo-rag-eval
git clone https://github.com/catonmoon/mo-rag-eval.git /opt/mo-rag-eval
cd /opt/mo-rag-eval
python3 -m venv venv
venv/bin/pip install -r requirements.lock.txt
```

## 3. Данные

Распаковать архив с данными в корень проекта — он содержит папку `data/`:

```bash
unzip mo-rag-eval-data-*.zip -d /opt/mo-rag-eval
ls data/gold/answers_gold.jsonl     # активный эталон — обязателен
```

Должно получиться:

```
data/gold/answers_gold.jsonl      # активный эталон {id, question, answer, context}
data/gold/versions/               # архив версий эталона + diff
data/questions/questions_public.jsonl
data/documents.zip                # исходные документы (кнопка Downloads)
data/human_overrides.json         # ручные вердикты (можно не переносить — начнётся с нуля)
```

`leaderboard.csv`, `eval_details.jsonl`, `data/judge_cache.json` создадутся сами при первом сабмите.

## 4. Переменные окружения

| Переменная | Обязательна | Значение |
|---|---|---|
| `OPENROUTER_API_KEY` | да | ключ OpenRouter |
| `EDIT_PASSWORDS` | **да** | пароль(и) входа через запятую. В коде заглушка `pass` — **сменить** |
| `EVAL_MODEL` | нет | модель судьи, по умолчанию `deepseek/deepseek-v4.1-flash` |
| `SECONDARY_MODEL` | нет | второй судья; по умолчанию `minimax/minimax-m3.1-flash` (используется, только если в UI включена галка Secondary) |
| `EVAL_CONCURRENCY` | нет | параллельных запросов к судье, по умолчанию 5 |
| `PROXY_URL` | нет | прокси до OpenRouter (`http://...` или `socks5://...`) |
| `GOLD_PATH_LOCAL` | нет | путь к активному эталону (по умолчанию `data/gold/answers_gold.jsonl`) |
| `QUESTIONS_PATH` | нет | по умолчанию `data/questions/questions_public.jsonl` |
| `DOCUMENTS_ZIP_PATH` | нет | по умолчанию `data/documents.zip` |
| `JUDGE_CACHE_PATH` | нет | по умолчанию `data/judge_cache.json` |
| `SHOW_DETAILS` | нет | `1` — показывать детали по вопросам |

Пробный запуск:

```bash
cd /opt/mo-rag-eval
OPENROUTER_API_KEY=sk-or-... EDIT_PASSWORDS=новый_пароль SHOW_DETAILS=1 venv/bin/python app.py
# открыть http://<сервер>:7862
```

## 5. systemd

`/etc/systemd/system/mo-rag-eval.service`:

```ini
[Unit]
Description=MO Rag Evaluation
After=network-online.target

[Service]
WorkingDirectory=/opt/mo-rag-eval
ExecStart=/opt/mo-rag-eval/venv/bin/python app.py
Environment=OPENROUTER_API_KEY=sk-or-...
Environment=EDIT_PASSWORDS=новый_пароль
Environment=EVAL_MODEL=deepseek/deepseek-v4.1-flash
Environment=SECONDARY_MODEL=minimax/minimax-m3.1-flash
Environment=SHOW_DETAILS=1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mo-rag-eval
journalctl -u mo-rag-eval -f
```

Ключ лучше вынести в `EnvironmentFile=/etc/mo-rag-eval.env` с правами `600`.

## 6. nginx + HTTPS (опционально)

`/etc/nginx/conf.d/mo-rag-eval.conf`:

```nginx
server {
    server_name eval.example.com;
    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:7862;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_http_version 1.1;
        proxy_read_timeout 3600;   # оценка идёт долго
        proxy_buffering off;
    }
    listen 80;
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d eval.example.com
```

Websocket-заголовки и `proxy_read_timeout` обязательны — без них ломается прогресс оценки.

## 7. Бэкапы — обязательно

Самое ценное — то, что правят люди через UI: **`data/gold/`** (эталоны и их версии) и
**`data/human_overrides.json`**. Плюс `leaderboard.csv` и `eval_details.jsonl` (история прогонов).

Пример cron (`crontab -e`), раз в час, хранение 30 дней:

```cron
0 * * * * cd /opt/mo-rag-eval && mkdir -p backups && tar czf backups/data_$(date +\%Y\%m\%d_\%H).tgz data/gold data/human_overrides.json leaderboard.csv && find backups -name 'data_*' -mtime +30 -delete
30 3 * * * cd /opt/mo-rag-eval && gzip -c eval_details.jsonl > backups/eval_details_$(date +\%Y\%m\%d).jsonl.gz && find backups -name 'eval_details_*' -mtime +30 -delete
```

Бэкап на том же диске не спасёт от потери сервера — копируйте `backups/` на другую машину
(rsync/rclone по cron).

## 8. Обновление кода

```bash
cd /opt/mo-rag-eval
git pull
venv/bin/pip install -r requirements.lock.txt
sudo systemctl restart mo-rag-eval
```

Данные в `data/` git не трогает (они в `.gitignore`).

## 9. Проверка

- Открыть сайт → ввести пароль → видна таблица лидерборда и список вопросов.
- Загрузить `example/example_submission.jsonl` → в логах `journalctl` идут вызовы судьи,
  по завершении **🔄 Refresh** показывает строку в лидерборде.
- Ошибки судьи в логах: `[judge:<model>] error ...` — обычно неверный ключ или модель.
