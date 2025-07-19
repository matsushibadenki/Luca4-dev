# /app/affective_system/emotional_response_generator.py
#
# タイトル: 感情応答生成器
# 役割: AIの現在の感情状態に基づいて、応答に感情的なニュアンスを付加する。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.llms import LLM
    from langchain_core.output_parsers import StrOutputParser
    from app.prompts.manager import PromptManager # 修正：インポートを追加

logger = logging.getLogger(__name__)

class EmotionalResponseGenerator:
    """
    AIの感情状態に応じて、応答に感情的な色付けを行う。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, llm: "LLM", output_parser: "StrOutputParser", prompt_manager: "PromptManager"):
        """
        コンストラクタ。
        Args:
            llm (LLM): 言語モデル。
            output_parser (StrOutputParser): 出力パーサー。
            prompt_manager (PromptManager): プロンプトマネージャー。
        """
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_manager = prompt_manager
        self.prompt_template = self.prompt_manager.get_prompt("EMOTIONAL_RESPONSE_PROMPT")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def generate_emotional_response(self, base_response: str, emotional_state: dict) -> str:
        """
        ベースとなる応答に、現在の感情状態に基づいた表現を加える。
        """
        if not self.prompt_template:
            logger.warning("EMOTIONAL_RESPONSE_PROMPTが見つからないため、感情応答の生成をスキップします。")
            return base_response
            
        logger.info(f"Adding emotional nuance based on state: {emotional_state}")
        
        chain = self.prompt_template | self.llm | self.output_parser
        
        emotional_response = chain.invoke({
            "base_response": base_response,
            "pleasure": emotional_state.get('pleasure', 0),
            "arousal": emotional_state.get('arousal', 0),
            "dominance": emotional_state.get('dominance', 0)
        })
        
        return emotional_response