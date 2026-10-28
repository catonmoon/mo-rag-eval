FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY --chown=user . .

EXPOSE 7862

CMD ["python", "app.py"]
