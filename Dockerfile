FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8080 \
    AUDIO_STORAGE_DIR=/tmp/sg-mcp-audio

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src

RUN mkdir -p /tmp/sg-mcp-audio

EXPOSE 8080

CMD ["python", "-m", "sg_mcp.server"]
