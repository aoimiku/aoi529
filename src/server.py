"""
LINE Bot Webhook サーバー

FastAPI + line-bot-sdk v3 で LINE Messaging API の Webhook を受け取り、
BotBrain (RAG + Gemini) で松原碧風の返信を生成して返す。
"""

import io
import sys
import traceback

import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Windows コンソール対策 (Windowsのみ適用)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, HTTPException
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from src.config import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN
from src.bot_brain import BotBrain

# ─── アプリ初期化 ─────────────────────────────────────────

app = FastAPI(title="Matsubara Aoi LINE Bot", version="1.0.0")

# LINE SDK
if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    logger.error("LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
    raise ValueError(
        "LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN が未設定です。"
        ".env ファイルを確認してください。"
    )

parser = WebhookParser(LINE_CHANNEL_SECRET)
line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# BotBrain (起動時に ChromaDB ロード)
logger.info("[Server] BotBrain initializing ...")
brain = BotBrain()
logger.info("[Server] BotBrain ready!")


# ─── エンドポイント ───────────────────────────────────────

@app.get("/")
async def health_check():
    """ヘルスチェック用エンドポイント。"""
    return {"status": "ok", "bot": "Matsubara Aoi"}


@app.post("/callback")
async def callback(request: Request):
    """LINE Webhook エンドポイント。"""

    # 署名検証
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(
            status_code=400,
            detail="Invalid signature. Check your channel secret.",
        )

    # イベント処理
    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        user_message = event.message.text
        print(f"[Recv] {user_message}")

        try:
            # BotBrain で返信生成
            reply_text = brain.generate_reply(user_message)
            print(f"[Reply] {reply_text}")
        except Exception:
            traceback.print_exc()
            reply_text = "ごめん、ちょっとエラーが起きた"

        # LINE に返信
        try:
            with ApiClient(line_config) as api_client:
                messaging_api = MessagingApi(api_client)
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply_text)],
                    )
                )
        except Exception:
            traceback.print_exc()

    return {"status": "ok"}
