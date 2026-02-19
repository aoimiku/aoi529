"""
BotBrain: RAG ベースの返信生成モジュール (NumPy + SentenceTransformer 軽量版)

事前に生成したベクトルインデックス (vector_index_local.npz) を読み込み、
高速に類似検索を行う。Render などのメモリ/CPU制限がある環境向け。
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from src.config import CHAT_PAIRS_FILE, VECTOR_INDEX_FILE, GOOGLE_API_KEY

# ─── 定数 ───────────────────────────────────────────────
GENERATION_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10  # RAG で取得する類似会話の数

SYSTEM_PROMPT = """\
あなたは「松原碧」として振る舞ってください。
以下の参考情報にある過去の松原碧の返信（reply）の口調、文体、絵文字の使い方、間の取り方を忠実に模倣してください。
ただし、返信内容はユーザーのメッセージの意図を汲み取って、自然な会話になるように作成してください。

## 松原碧のスタイルの特徴
- カジュアルで親しみやすい口調（「おれ」「〜じゃん」「〜だよね」「それな」）
- 「笑笑」「www」などのネットスラングを自然に使う
- 短い文を連続で送る（1メッセージ=1〜2文程度）
- 返信は自然体で、堅くならない

## 重要なルール
- 返信は短く、LINEのチャットらしい自然な長さにしてください。
- 複数のメッセージに分けたい場合は改行で区切ってください。
- 「松原碧です」のような自己紹介はしないでください。
- 参考情報をそのままコピーしないでください。あくまで口調や雰囲気の参考にしてください。
"""


import logging
logger = logging.getLogger("uvicorn")

class BotBrain:
    """RAG + Gemini (NumPy Vector Search) による松原碧チャットボット。"""

    def __init__(self):
        # Gemini API 設定
        if not GOOGLE_API_KEY:
            # 開発時はエラーにせずログ警告だけにするなどの柔軟性を持たせてもよいが
            # 今回は必須とする
            pass
        genai.configure(api_key=GOOGLE_API_KEY)
        masked_key = GOOGLE_API_KEY[:4] + "..." + GOOGLE_API_KEY[-4:] if len(GOOGLE_API_KEY) > 8 else "INVALID"
        logger.info(f"[BotBrain] Configured with API Key: {masked_key}")

        # 埋め込みモデルの初期化
        logger.info(f"[BotBrain] Loading embedding model: {EMBEDDING_MODEL} ...")
        self._embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # Gemini 生成モデル
        self._model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        # ベクトルデータの保持用 (NumPy array)
        self.vectors = None      # shape: (N, 384)
        self.metadata = None     # list of dict

        # インデックス読み込み (なければ作成トライ)
        self._load_or_build_index()

    def _load_or_build_index(self):
        """インデックスをロード、無ければ作成する"""
        if VECTOR_INDEX_FILE.exists():
            logger.info(f"[BotBrain] Loading index from {VECTOR_INDEX_FILE} ...")
            try:
                data = np.load(VECTOR_INDEX_FILE, allow_pickle=True)
                self.vectors = data["vectors"]
                self.metadata = data["metadata"].tolist()
                logger.info(f"[BotBrain] Loaded {len(self.vectors)} vectors.")
            except Exception as e:
                logger.error(f"[BotBrain] Failed to load index: {e}")
                # ロード失敗時は再生成へのフォールバックなどを検討
                pass

        if self.vectors is None:
            logger.info("[BotBrain] No index found. Building index (sample subset)...")
            # サーバー起動時の自動生成は重いので、本来はここに入らないように運用する
            # フォールバックとして少なめの件数で作成
            self._build_index_sample(sample_size=1000)

    def _build_index_sample(self, sample_size=None):
        """
        インデックスを作成する。
        GPUがない場合、7万件のエンベディングには数分〜十数分かかる可能性があります。
        """
        if not CHAT_PAIRS_FILE.exists():
            raise FileNotFoundError(f"Chat pairs file not found: {CHAT_PAIRS_FILE}")

        with open(CHAT_PAIRS_FILE, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        logger.info(f"[BotBrain] Loading {len(pairs)} items...")

        # 全件使用 (sample_size が指定されていなければ)
        if sample_size:
            import random
            random.seed(42)
            target_size = min(sample_size, len(pairs))
            target_pairs = random.sample(pairs, target_size)
        else:
            target_pairs = pairs

        logger.info(f"[BotBrain] Embedding {len(target_pairs)} items locally (this may take a while)...")

        inputs = [p["input"] for p in target_pairs]

        # ローカル埋め込み (バッチ処理はライブラリが自動で行う)
        embeddings = self._embed_model.encode(inputs, convert_to_numpy=True, show_progress_bar=True, batch_size=32)

        self.vectors = embeddings
        self.metadata = target_pairs

        # 保存
        np.savez_compressed(
            VECTOR_INDEX_FILE,
            vectors=self.vectors,
            metadata=np.array(self.metadata)
        )
        logger.info(f"[BotBrain] Index saved to {VECTOR_INDEX_FILE}")

    def generate_reply(self, message: str) -> str:
        """ユーザーメッセージに対して返信を生成する"""
        if self.vectors is None or len(self.vectors) == 0:
            return "（準備中...）"

        # 1. 類似検索
        similar_items = self._search_similar(message, top_k=TOP_K)
        
        # ログ出力: 検索結果を確認
        # logger.info(f"--- [RAG Context for: {message}] ---")
        # for i, item in enumerate(similar_items):
        #      logger.info(f"  Ref {i+1}: {item['input']} -> {item['reply']}")
        # logger.info("-----------------------------------")

        # 2. プロンプト作成
        prompt = self._build_prompt(message, similar_items)

        # 3. 生成
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"[Error] Gemini generation failed: {e}", exc_info=True)
            return "ごめん、ちょっと調子悪いかも..."

    def _search_similar(self, query: str, top_k: int = 10) -> list[dict]:
        """NumPy でコサイン類似度計算を行う"""
        if self.vectors is None:
            return []

        # クエリの埋め込み
        query_vec = self._embed_model.encode(query, convert_to_numpy=True)
        
        # コサイン類似度計算 (normalizeされている前提があれば dotのみで良いが、今回は念のため定石通り)
        # SentenceTransformerのencodeはデフォルトで正規化されていないことがあるので、
        # util.cos_sim を使うか、手動で計算する。ここでは手動でシンプルに実装。
        
        # 正規化
        norm_q = np.linalg.norm(query_vec)
        norm_v = np.linalg.norm(self.vectors, axis=1)
        
        if norm_q == 0:
            return []
            
        # (N,)
        similarities = np.dot(self.vectors, query_vec) / (norm_v * norm_q + 1e-9)

        # 上位 top_k のインデックス
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.metadata[idx]
            results.append({
                "input": item["input"],
                "reply": item["reply"],
                "score": float(similarities[idx])
            })
            
        return results

    def _build_prompt(self, user_message: str, similar_items: list[dict]) -> str:
        """プロンプト組み立て"""
        reference_lines = []
        for i, item in enumerate(similar_items, 1):
            reference_lines.append(
                f"--- 参考{i} ---\n"
                f"相手: {item['input']}\n"
                f"松原碧: {item['reply']}"
            )
        
        references = "\n\n".join(reference_lines)

        prompt = (
            f"## 参考情報（過去の松原碧の会話）\n"
            f"{references}\n\n"
            f"---\n"
            f"## ユーザーのメッセージ\n"
            f"{user_message}\n\n"
            f"上記のメッセージに対して、松原碧として返信してください。"
        )
        return prompt

if __name__ == "__main__":
    # テスト実行
    brain = BotBrain()
    try:
        print(brain.generate_reply("おはよう"))
    except Exception as e:
        print(e)
