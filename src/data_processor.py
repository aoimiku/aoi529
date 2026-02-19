"""
LINE チャット履歴 → 学習用会話ペア変換スクリプト

data/line_history.txt を読み込み、「松原碧」の発言を reply とした
会話ペア (input/reply) を data/chat_pairs.json に出力する。
2024年11月1日以降のデータのみ抽出する。
"""

import io
import sys

# Windows コンソール (cp932) でのエンコードエラーを防止
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import re
from datetime import date
from pathlib import Path

# ─── 定数 ───────────────────────────────────────────────
TARGET_USER = "松原碧"
CUTOFF_DATE = date(2024, 11, 1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "line_history.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "chat_pairs.json"

# 日付ヘッダー: 2024/11/01(金)
DATE_HEADER_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})\(.+\)\s*$")

# メッセージ行: HH:MM\tユーザー名\tメッセージ
MESSAGE_RE = re.compile(r"^(\d{1,2}:\d{2})\t(.+?)\t(.+)$")

# ─── フィルタ条件 ────────────────────────────────────────
# システムメッセージ（通話系・アルバム系など）
SYSTEM_PATTERNS = [
    "☎ 通話時間",
    "☎ 通話をキャンセルしました",
    "☎ 不在着信",
    "☎ 通話に応答がありませんでした",
    "アルバムを作成しました",
    "アルバムに追加しました",
    "ノートに投稿しました",
    "イベントを作成しました",
    "イベントに参加",
    "グループに招待",
    "トーク履歴",
    "メッセージの送信を取り消しました",
    "が参加しました",
    "が退出しました",
]

# ノイズ除去: スタンプ・写真・動画のみの行
NOISE_ONLY_RE = re.compile(r"^\[(?:スタンプ|写真|動画|ファイル|連絡先)\]$")

# URL を含む行
URL_RE = re.compile(r"https?://")


def is_system_message(text: str) -> bool:
    """通話キャンセル・アルバム作成などのシステムメッセージか判定する。"""
    return any(pattern in text for pattern in SYSTEM_PATTERNS)


def is_noise(text: str) -> bool:
    """学習に不要なノイズ行か判定する。"""
    if NOISE_ONLY_RE.match(text):
        return True
    if URL_RE.search(text):
        return True
    return False


def parse_chat_file(filepath: Path) -> list[dict]:
    """
    LINE エクスポートテキストを解析し、メッセージのリストを返す。

    Returns:
        [{"user": str, "text": str, "date": date}, ...]
    """
    messages: list[dict] = []
    current_date: date | None = None
    in_range = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")

            # 日付ヘッダーの検出
            dm = DATE_HEADER_RE.match(line)
            if dm:
                y, m, d = int(dm.group(1)), int(dm.group(2)), int(dm.group(3))
                current_date = date(y, m, d)
                in_range = current_date >= CUTOFF_DATE
                continue

            # 日付が範囲外ならスキップ
            if not in_range:
                continue

            # メッセージ行の検出
            mm = MESSAGE_RE.match(line)
            if mm:
                user = mm.group(2)
                text = mm.group(3).strip()

                # システムメッセージを除外
                if is_system_message(text):
                    continue

                # ノイズを除外
                if is_noise(text):
                    continue

                messages.append({
                    "user": user,
                    "text": text,
                    "date": current_date,
                })

    return messages


def build_chat_pairs(messages: list[dict]) -> list[dict]:
    """
    メッセージリストから会話ペアを作成する。

    ロジック:
      - 「相手」の連続発言をまとめて input にする。
      - その直後の「松原碧」の連続発言をまとめて reply にする。
      - 1つの会話ペアができたらリセットし、次のペアに進む。
    """
    pairs: list[dict] = []
    other_buffer: list[str] = []   # 相手側の発言バッファ
    target_buffer: list[str] = []  # 松原碧の発言バッファ

    for msg in messages:
        if msg["user"] != TARGET_USER:
            # 相手の発言
            if target_buffer:
                # 松原碧の発言が溜まっていれば、前のペアを確定
                if other_buffer:
                    pairs.append({
                        "input": "\n".join(other_buffer),
                        "reply": "\n".join(target_buffer),
                    })
                other_buffer = []
                target_buffer = []
            other_buffer.append(msg["text"])
        else:
            # 松原碧の発言
            target_buffer.append(msg["text"])

    # 最後に残ったペアを処理
    if other_buffer and target_buffer:
        pairs.append({
            "input": "\n".join(other_buffer),
            "reply": "\n".join(target_buffer),
        })

    return pairs


def main():
    print(f"[FILE] Input: {INPUT_FILE}")
    print(f"[DATE] Period: {CUTOFF_DATE} or later")
    print(f"[USER] Target: {TARGET_USER}")
    print()

    if not INPUT_FILE.exists():
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    # 1. Parse
    messages = parse_chat_file(INPUT_FILE)
    print(f"[OK] Filtered messages: {len(messages):,}")

    # User breakdown
    users = {}
    for msg in messages:
        users[msg["user"]] = users.get(msg["user"], 0) + 1
    for user, count in users.items():
        print(f"   - {user}: {count:,}")

    # 2. Build pairs
    pairs = build_chat_pairs(messages)
    print(f"[OK] Chat pairs: {len(pairs):,}")

    # 3. JSON output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Output: {OUTPUT_FILE}")

    # Show samples
    if pairs:
        print("\n--- Sample (first 3) ---")
        for i, pair in enumerate(pairs[:3], 1):
            print(f"\n[{i}]")
            print(f"  input: {pair['input'][:80]}")
            print(f"  reply: {pair['reply'][:80]}")


if __name__ == "__main__":
    main()
