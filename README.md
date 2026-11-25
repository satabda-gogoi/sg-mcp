# sg-mcp — Cloud Gemini 3.8 Flash TTS MCP Server

A cloud-ready **Model Context Protocol (MCP)** server built with **FastMCP 4 (`streamable-http`)** and the official **Google GenAI SDK (`google-genai`)**. It exposes expressive text-to-speech tools powered by **`gemini-3.8-flash-tts`** with automatic multi-API-key failover, full voice discovery (**2,150+ male, female, and neutral voices**), and direct **downloadable `.wav` links** designed for **Gemini Business / Gemini Enterprise custom connections** and modern MCP clients.

---

## Features

- **Streamable HTTP Transport (`/mcp`)**: Stateless HTTP transport compatible with **Gemini Enterprise / Business Custom MCP Connections**, Cursor, Claude Desktop, and FastMCP clients.
- **`generate_voice` Tool**:
  - Uses **`gemini-3.8-flash-tts`**.
  - `voice` is optional — defaults to male voice **`charon`** if omitted.
  - Supports optional `style` (e.g., `"warm and authoritative"`, `"cheerful and upbeat"`), `language_code` (e.g., `"en-US"`, `"hi-IN"`), `temperature`, `seed`, and `filename_prefix`.
  - Returns **both** a direct downloadable `.wav` URL (`/download/<id>/<file>.wav`), browser preview URL (`?inline=1`), structured JSON metadata, and inline MCP `AudioContent`.
- **`list_voices` Tool**:
  - Fetches and caches all **2,150+ Gemini TTS voices** across `male`, `female`, and `neutral` genders and 29+ languages/accents (with a built-in 30-voice flagship fallback).
  - Filterable by `gender`, `language_code`, `pitch`, `featured_only`, and keyword `search`.
- **Automatic Multi-Key Failover**:
  - Set `GEMINI_API_KEYS=key1,key2` (or `GEMINI_API_KEY` + `GEMINI_API_KEY_2`). If Key 1 hits quota/rate limits or fails, requests automatically fail over to Key 2.

---

## Project Structure

```text
├── src/sg_mcp/
│   ├── config.py        # Environment settings & multi-key parsing
│   ├── voices.py        # 2,150+ voice catalog + flagship fallback + resolution
│   ├── tts_service.py   # gemini-3.8-flash-tts synthesis & key failover
│   ├── storage.py       # Ephemeral .wav storage with TTL & max-file eviction
│   └── server.py        # FastMCP Streamable HTTP app + /health + /download routes
├── tests/
│   └── test_server.py   # Unit & HTTP integration tests
├── Dockerfile
├── render.yaml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Local Setup & Running

1. **Create virtual environment & install dependencies** (Python 3.12+):
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
   ```
   *(Or with standard `pip`: `python -m venv .venv && pip install -r requirements.txt`)*

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Gemini API keys:
   ```env
   GEMINI_API_KEYS=your_primary_key,your_backup_key
   ```
   > **Security:** Never commit `.env` or real API keys to GitHub. `.gitignore` already excludes `.env`.

3. **Run the server**:
   ```bash
   .venv/Scripts/python.exe -m sg_mcp.server
   ```
   - Health check: `http://127.0.0.1:8080/health`
   - MCP endpoint: `http://127.0.0.1:8080/mcp`

4. **Run tests**:
   ```bash
   .venv/Scripts/pytest.exe -v
   ```

---

## Free Cloud Deployment

### Option 1: Google Cloud Run (Recommended — Always Free Tier)
Google Cloud Run has an **Always Free tier** (2 million requests/month, 360,000 vCPU-seconds, 180,000 GiB-seconds) and is Google's recommended host for **Gemini Enterprise / Business** custom MCP servers.

1. Install [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) and authenticate:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```
2. Enable required APIs:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
3. Deploy directly from source:
   ```bash
   gcloud run deploy sg-mcp \
     --source . \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 512Mi \
     --min-instances 0 \
     --max-instances 2 \
     --set-env-vars "GEMINI_TTS_MODEL=gemini-3.8-flash-tts,DEFAULT_VOICE_ID=charon,AUDIO_STORAGE_DIR=/tmp/sg-mcp-audio" \
     --set-env-vars "GEMINI_API_KEYS=KEY_1,KEY_2"
   ```
   *(For stricter security, store `GEMINI_API_KEYS` in Google Secret Manager and mount via `--set-secrets`.)*
4. Copy the service URL printed by Cloud Run (e.g., `https://sg-mcp-xxxxxx-uc.a.run.app`).
   - Your MCP URL is: `https://sg-mcp-xxxxxx-uc.a.run.app/mcp`
   - Optional: set `PUBLIC_BASE_URL=https://sg-mcp-xxxxxx-uc.a.run.app` on Cloud Run so download URLs always use your canonical HTTPS domain.

---

### Option 2: Render (Free Tier — No Credit Card Required)
Render offers a **Free Web Service** tier for Docker apps (note: free instances sleep after 15 minutes of inactivity and take ~30–50s to wake on the first request).

1. Push this repo to GitHub (`https://github.com/satabda-gogoi/sg-mcp`).
2. Log in to [Render](https://render.com), click **New +** -> **Blueprint** (or **Web Service**), and select your GitHub repository (`render.yaml` is already included).
3. In Render's Environment Variables prompt, enter:
   - `GEMINI_API_KEYS`: `your_key_1,your_key_2`
   - *(Optional)* `MCP_AUTH_TOKEN`: a secret bearer token if you want to restrict `/mcp` access.
4. Click **Apply / Deploy**. Once live, your MCP URL will be:
   `https://<your-service>.onrender.com/mcp`

---

## Connecting to Gemini Business / Gemini Enterprise

1. Open the **Gemini Enterprise / Gemini Business Admin Console** -> **Data Stores / Connections / Custom Connectors**.
2. Add a **Custom MCP Server** connector:
   - **Transport**: **Streamable HTTP** (SSE is not supported by Gemini Enterprise).
   - **MCP Endpoint URL**: `https://<your-deployed-domain>/mcp`
   - **Authentication**: None (or Bearer Token if you set `MCP_AUTH_TOKEN`).
3. Refresh/load actions and enable:
   - `generate_voice`
   - `list_voices`
4. Attach the connector to your Gemini Business app/assistant. You can now ask Gemini:
   - *"List available female voices with a warm tone"*
   - *"Generate a voice script saying 'Welcome to our Q4 briefing' using the default male voice"*
   - *"Generate this script in Puck's upbeat voice with an energetic style"*
