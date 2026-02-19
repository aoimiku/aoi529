"""
設定管理モジュール
環境変数を .env ファイルから読み込み、アプリ全体で参照できるようにする。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# プロジェクトルートの .env を読み込む
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── Google Gemini API ─────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ─── LINE Messaging API ───────────────────────────────
LINE_CHANNEL_SECRET: str = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

# ─── パス ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
CHAT_PAIRS_FILE = DATA_DIR / "chat_pairs.json"
VECTOR_INDEX_FILE = DATA_DIR / "vector_index_local.npz"
