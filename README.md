# sg-mcp — Cloud Gemini 3.8 Flash TTS & YouTube MCP Server

A cloud-ready **Model Context Protocol (MCP)** server built with **FastMCP 4 (`streamable-http`)**, **Google GenAI SDK (`google-genai`)**, and **YouTube Data API v3 (`google-api-python-client`)**. It powers expressive voice generation via **`gemini-3.8-flash-tts`** and full YouTube upload/analytics tools, built for **Gemini Business / Gemini Enterprise custom connections** and modern MCP clients.

---

## Features

### 1. Voice Generation (`gemini-3.8-flash-tts`)
- **`generate_voice`**: Generates `.wav` speech from a script. `voice` is optional (defaults to male voice **`charon`**). Supports optional `style`, `language_code`, `temperature`, `seed`, and `filename_prefix`. Returns a direct downloadable `.wav` URL, browser preview URL (`?inline=1`), structured metadata, and inline MCP `AudioContent`.
- **`list_voices`**: Searches and filters **2,150+ Gemini TTS voices** across `male`, `female`, and `neutral` genders and 29+ languages (with a 30-voice flagship fallback).
- **Multi-Key Failover**: `GEMINI_API_KEYS=key1,key2` automatically fails over if Key 1 hits quota or errors.

### 2. YouTube Upload, Stats & Comments (YouTube Data API v3)
- **`upload_youtube_video`**: Uploads **YouTube Shorts** (`video_type="short"`, adds `#Shorts` tag/marker) or **long-form videos** (`video_type="long"`) from a local file path or public HTTP/HTTPS video URL. Supports `privacy_status` (`private`, `unlisted`, `public`), `tags`, `category_id`, `made_for_kids`, scheduled `publish_at`, and optional custom `thumbnail_source`.
- **`get_youtube_stats`**: Given a `video_id` or YouTube URL, returns views, **total likes**, **total comments**, duration, and privacy status. When `video_id` is omitted, returns channel subscriber count, total views, total videos, and recent videos with their stats.
- **`get_youtube_comments`**: Reads top-level comments and nested replies for a `video_id`/URL or across the whole channel, with sorting (`relevance`/`time`), pagination, and `fetch_all` mode.
- **Built-in OAuth Setup Routes**: `/youtube/auth` and `/youtube/callback` let you authorize your YouTube channel in one browser click and obtain `YOUTUBE_REFRESH_TOKEN` for cloud hosting.

---

## Local Setup & Running

1. **Create virtual environment & install dependencies** (Python 3.12+):
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
   ```
2. **Configure `.env`**:
   ```bash
   cp .env.example .env
   ```
   Fill in `GEMINI_API_KEYS`, `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, and (after the 1-time OAuth step below) `YOUTUBE_REFRESH_TOKEN`.
   > **Security:** Never commit `.env` or `youtube_token.json` to GitHub. Both are excluded in `.gitignore`.

3. **Run the server**:
   ```bash
   .venv/Scripts/python.exe -m sg_mcp.server
   ```
   - Status: `http://127.0.0.1:8080/`
   - Health: `http://127.0.0.1:8080/health`
   - MCP endpoint: `http://127.0.0.1:8080/mcp`

4. **One-Time YouTube OAuth Setup**:
   1. In **Google Cloud Console** -> your project -> **APIs & Services -> Library**, ensure **YouTube Data API v3** is enabled.
   2. Under **APIs & Services -> Credentials -> your OAuth 2.0 Web Client**, add Authorized redirect URIs:
      - Local: `http://localhost:8080/youtube/callback`
      - Cloud (Render): `https://<your-service>.onrender.com/youtube/callback`
   3. Under **OAuth consent screen -> Test users** (if app is in Testing mode), add your channel email.
   4. Start the server and open `http://localhost:8080/youtube/auth` (or `https://<your-service>.onrender.com/youtube/auth`) in your browser and approve access.
   5. Copy the printed `YOUTUBE_REFRESH_TOKEN` into `.env` (for local) and into Render / Cloud Run Environment Variables (for cloud).

---

## Will GitHub Commits Reflect Automatically in Render & Gemini MCP?

- **Render:** **Yes** — if Auto-Deploy is enabled on your Render service (the default when connected to GitHub), pushing to `main` automatically triggers a rebuild and redeploy.
  - **Important:** When you add *new* environment variables (such as `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, and `YOUTUBE_REFRESH_TOKEN`), you must add them in **Render Dashboard -> Environment -> Save Changes**, because secrets are never committed to GitHub.
- **Gemini Business / Gemini Enterprise Custom MCP Connection:** **No, not automatically for newly added tools.**
  - Existing tools keep working, but Gemini Enterprise caches the tool list (`actions`) on the data store and disables new actions by default.
  - After Render finishes deploying, open **Google Cloud Console -> Gemini Enterprise -> Data stores -> your Custom MCP Server -> Actions tab -> click "Reload custom actions"**, then check and **Enable** the new actions (`upload_youtube_video`, `get_youtube_stats`, `get_youtube_comments`). If a chat session still doesn't see them, toggle the connector off/on and hard-refresh (`Ctrl+F5`).
