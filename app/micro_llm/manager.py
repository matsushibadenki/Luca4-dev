# /app/micro_llm/manager.py
#
# タイトル: MicroLLMマネージャー
# 役割: 生成された専門家MicroLLMのライフサイクル（保存、読み込み、一覧表示）を管理する。

from __future__ import annotations
import os
import logging
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from app.llm_providers.base import LLMProvider
    from .creator import MicroLLMCreator

logger = logging.getLogger(__name__)

class MicroLLMManager:
    """
    専門家MicroLLMの管理を行うクラス。
    """
    def __init__(self, llm_provider: "LLMProvider", creator: "MicroLLMCreator"):
        self.llm_provider = llm_provider
        self.creator = creator

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def list_specialized_models(self) -> List[str]:
        """
        現在利用可能な専門家MicroLLMのリストを返す。
        Ollamaのモデルリストから、特定の命名規則を持つものをフィルタリングする。
        """
        try:
            all_models = self.llm_provider.list_models()
            if not all_models or "models" not in all_models:
                return []
            
            # "micro-llm-" プレフィックスを持つモデルのみを専門家モデルとみなす
            specialized_models = [
                model["name"] for model in all_models["models"] 
                if model.get("name", "").startswith("micro-llm-")
            ]
            return specialized_models
        except Exception as e:
            logger.error(f"専門家モデルのリスト取得中にエラーが発生しました: {e}", exc_info=True)
            return []
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def create_and_load_expert(self, topic: str, documents: List[str]) -> str:
        """
        指定されたトピックとドキュメントに基づいて、新しい専門家MicroLLMを作成し、ロードする。
        """
        model_name = self.creator.create_model_from_documents(topic, documents)
        if model_name:
            logger.info(f"専門家MicroLLM '{model_name}' が正常に作成されました。")
            # ここではモデル名を返すだけだが、将来的にはロード処理もここで行う
            return model_name
        else:
            logger.error(f"トピック '{topic}' の専門家MicroLLMの作成に失敗しました。")
            return ""