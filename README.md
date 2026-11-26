# sg-mcp — Cloud Gemini 3.8 Flash TTS, YouTube & Instagram MCP Server

A cloud-ready **Model Context Protocol (MCP)** server built with **FastMCP 4 (`streamable-http`)**, **Google GenAI SDK (`google-genai`)**, **YouTube Data API v3**, and **Meta / Instagram Graph API (`v27.0`)**, built for **Gemini Business / Gemini Enterprise custom connections** and modern MCP clients.

---

## Features & Tools

### 1. Voice Generation (`gemini-3.8-flash-tts`)
- **`generate_voice`**: Generates `.wav` speech from a script. `voice` is optional (defaults to male voice **`charon`**). Supports optional `style`, `language_code`, `temperature`, `seed`, and `filename_prefix`. Returns a direct downloadable `.wav` URL, browser preview URL (`?inline=1`), structured metadata, and inline MCP `AudioContent`.
- **`list_voices`**: Searches and filters **2,150+ Gemini TTS voices** across `male`, `female`, and `neutral` genders and 29+ languages (with a 30-voice flagship fallback).
- **Multi-Key Failover**: `GEMINI_API_KEYS=key1,key2` automatically fails over if Key 1 hits quota or errors.

### 2. YouTube (YouTube Data API v3)
- **`upload_youtube_video`**: Uploads **YouTube Shorts** (`video_type="short"`) or **long-form videos** (`video_type="long"`) from a local file path, public HTTP/HTTPS URL, or `video_base64`.
- **`get_youtube_stats`**: Returns video views, total likes, total comments, duration, and privacy status (or channel subscribers + recent videos when `video_id` is omitted).
- **`get_youtube_comments`**: Reads top-level comments and nested replies for a video or across the channel.
- **OAuth Routes**: `/youtube/auth` and `/youtube/callback`.

### 3. Instagram (Instagram Graph API `v27.0`)
- **`upload_instagram_content`**: Publishes **Reels**, **Photos**, **Stories**, or **Carousels** (`2–10` items) to your Instagram Professional (Creator/Business) account. Accepts a public HTTPS `media_url` (or `media_base64` / local path staged via the server's public `/download/...` HTTPS URL).
- **`get_instagram_stats`**: Reads account followers, media count, recent posts (with `like_count` and `comments_count`), and post/account insights (`views`, `reach`, `saved`, `shares`, `total_interactions`).
- **`get_instagram_comments`**: Reads comments and replies on a specific `media_id` or across recent posts.
- **OAuth Routes**: `/instagram/auth` and `/instagram/callback` (supports both `INSTAGRAM_AUTH_MODE=facebook` via Facebook Login + linked Facebook Page, and `INSTAGRAM_AUTH_MODE=instagram` via Instagram Login).

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
   > **Security:** Never commit `.env`, `youtube_token.json`, `instagram_token.json`, passwords, or API keys to GitHub. They are excluded in `.gitignore`.
3. **Run the server**:
   ```bash
   .venv/Scripts/python.exe -m sg_mcp.server
   ```
   - Status: `http://127.0.0.1:8080/`
   - Health: `http://127.0.0.1:8080/health`
   - MCP endpoint: `http://127.0.0.1:8080/mcp`
4. **Run tests**:
   ```bash
   .venv/Scripts/pytest.exe -v
   ```

---

## Meta / Instagram Setup (App ID `META_APP_ID`)

Meta's official Graph API **does not use an API key or your Instagram password** — it uses **App ID + App Secret + an OAuth Access Token**.

### Recommended: Instagram API with Unified Login (`INSTAGRAM_AUTH_MODE=unified`)
1. **In [Meta for Developers](https://developers.facebook.com) → Your App:**
   - Make sure your app is owned by a **Meta Business Portfolio** and has the **"Manage messaging & content on Instagram"** use case added.
   - Go to **Use cases → Manage messaging & content on Instagram → API setup with Unified login** and create/open your **Facebook Login for Business Configuration**.
   - Select the permissions (`instagram_basic`, `instagram_content_publish`, `instagram_manage_comments`, `instagram_manage_insights`, `business_management`) and copy the **Configuration ID (`META_CONFIG_ID`)**.
   - Under **App settings → Basic**, add `sg-mcp.onrender.com` to **App Domains** and add Website platform Site URL `https://sg-mcp.onrender.com/`.
   - Under **Facebook Login for Business → Settings → Valid OAuth Redirect URIs**, add:
     - `https://<your-render-service>.onrender.com/instagram/callback`
     - `http://localhost:8080/instagram/callback`
2. **Set Env Vars** in Render Dashboard → Environment (and `.env`):
   - `META_APP_ID`
   - `META_APP_SECRET`
   - `META_CONFIG_ID`
   - `INSTAGRAM_AUTH_MODE=unified`
3. **Authorize Once**:
   - Visit `https://<your-render-service>.onrender.com/instagram/auth`, sign in with Instagram or Facebook, select `@stupidlysimpleone`, and approve permissions.
   - Copy the displayed `INSTAGRAM_USER_ID` and `INSTAGRAM_ACCESS_TOKEN` into Render's Environment Variables (and `.env`).

*(Legacy `INSTAGRAM_AUTH_MODE=facebook` with a linked Facebook Page, and `INSTAGRAM_AUTH_MODE=instagram`, are also supported.)*

---

## Render & Gemini MCP Updates

- **Render**: Pushes to `main` auto-deploy code changes; whenever you add **new** environment variables (`META_APP_ID`, `META_APP_SECRET`, `INSTAGRAM_USER_ID`, `INSTAGRAM_ACCESS_TOKEN`), add them in **Render Dashboard → Environment → Save Changes**.
- **Gemini Business / Gemini Enterprise**: After Render finishes deploying, go to **Google Cloud Console → Gemini Enterprise → Data stores → your Custom MCP Server → Actions tab → Reload custom actions**, then select and **Enable** the new tools (`upload_instagram_content`, `get_instagram_stats`, `get_instagram_comments`).
