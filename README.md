# SentinelML AI Backend

A secure AI proxy backend for SentinelML V2. This allows the desktop application to use AI features without distributing the Gemini API key to client machines.

## Features
- Validates requests from SentinelML desktop client.
- Keeps Gemini API key secure on the server.
- Supports model fallback on high load/errors.
- IP-based rate limiting.
- Configurable CORS.
- Enforces message size limits.

## Setup
1. `pip install -r requirements.txt`
2. `cp .env.example .env` and add your `GEMINI_API_KEY`.
3. `uvicorn app.main:app --reload`
