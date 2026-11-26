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

Meta's official Graph API **does not use an API key or your Instagram password** — it uses **App ID + App Secret + an OAuth Access Token**:

1. **Link Instagram to a Facebook Page** *(required for `INSTAGRAM_AUTH_MODE=facebook`)*:
   - Ensure your Instagram account is a **Professional (Creator or Business)** account and is linked to a **Facebook Page** you administer.
   - *(Alternatively, add the **Instagram → API setup with Instagram login** product in Meta for Developers and set `INSTAGRAM_AUTH_MODE=instagram`, which does not require a Facebook Page.)*
2. **In [Meta for Developers](https://developers.facebook.com) → Your App:**
   - **Use cases / Permissions**: Add `instagram_basic`, `instagram_content_publish`, `instagram_manage_comments`, `instagram_manage_insights`, `pages_show_list`, `pages_read_engagement`.
   - **Facebook Login → Settings → Valid OAuth Redirect URIs**: Add:
     - `http://localhost:8080/instagram/callback`
     - `https://<your-render-service>.onrender.com/instagram/callback`
   - **App Roles**: Add your Facebook account as Admin/Developer while the app is in Development mode.
3. **Set Env Vars** in `.env` (and Render Dashboard → Environment):
   - `META_APP_ID`
   - `META_APP_SECRET`
   - `INSTAGRAM_AUTH_MODE=facebook` (or `instagram`)
4. **Authorize Once**:
   - Visit `http://localhost:8080/instagram/auth` (or `https://<your-render-service>.onrender.com/instagram/auth`), approve all permissions and select your Facebook Page + Instagram account.
   - Copy the displayed `INSTAGRAM_USER_ID` and `INSTAGRAM_ACCESS_TOKEN` into Render's Environment Variables (and `.env`).

---

## Render & Gemini MCP Updates

- **Render**: Pushes to `main` auto-deploy code changes; whenever you add **new** environment variables (`META_APP_ID`, `META_APP_SECRET`, `INSTAGRAM_USER_ID`, `INSTAGRAM_ACCESS_TOKEN`), add them in **Render Dashboard → Environment → Save Changes**.
- **Gemini Business / Gemini Enterprise**: After Render finishes deploying, go to **Google Cloud Console → Gemini Enterprise → Data stores → your Custom MCP Server → Actions tab → Reload custom actions**, then select and **Enable** the new tools (`upload_instagram_content`, `get_instagram_stats`, `get_instagram_comments`).
