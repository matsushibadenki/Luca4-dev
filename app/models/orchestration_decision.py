# /app/models/orchestration_decision.py
#
# タイトル: オーケストレーション決定モデル
# 役割: OrchestrationAgentが下す決定のデータ構造を定義する。

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class OrchestrationDecision(BaseModel):
    """
    OrchestrationAgentによって決定される、実行モードと関連する設定。
    """
    chosen_mode: str = Field(
        ...,
        description="選択された実行パイプラインの名前（例：'simple', 'full'）。"
    )
    reason: str = Field(
        ...,
        description="そのパイプラインが選択された理由。"
    )
    agent_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="パイプライン内の特定のエージェントに対する動的な設定オーバーライド。"
    )
    reasoning_emphasis: Optional[str] = Field(
        None,
        description="推論の強調点（例：'bird's_eye_view', 'detail_oriented'）。"
    )