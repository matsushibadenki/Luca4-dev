# /app/internal_dialogue/consciousness_staging_area.py
#
# タイトル: 意識のステージングエリア
# 役割: 内部対話における複数の思考や視点を一時的に保持し、統合のための準備をする。

from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .mediator_agent import MediatorAgent

class ConsciousnessStagingArea:
    """
    内部対話の参加者からの思考を集め、統合を待つための中央ハブ。
    """
    def __init__(self, llm: any, mediator_agent: "MediatorAgent"):
        self.llm = llm
        self.mediator_agent = mediator_agent
        self.staged_thoughts: List[str] = []

    def stage_thought(self, thought: str):
        """思考をステージングエリアに追加する。"""
        self.staged_thoughts.append(thought)

    def get_staged_thoughts(self) -> List[str]:
        """現在ステージングされているすべての思考を取得する。"""
        return self.staged_thoughts

    def clear_stage(self):
        """ステージングエリアをクリアする。"""
        self.staged_thoughts = []