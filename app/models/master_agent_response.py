# /app/models/master_agent_response.py
#
# タイトル: マスターエージェント応答モデル
# 役割: すべてのパイプラインが返す最終的な応答の標準的な形式を定義する。

from pydantic import BaseModel, Field
from typing import Optional

class MasterAgentResponse(BaseModel):
    """
    推論パイプラインの最終的な出力をカプセル化するPydanticモデル。
    """
    final_answer: str = Field(
        ...,
        description="ユーザーに対する最終的な、整形された応答。"
    )
    self_criticism: Optional[str] = Field(
        None,
        description="応答生成プロセスにおける自己評価や内省の結果。"
    )
    potential_problems: Optional[str] = Field(
        None,
        description="自己評価中に特定された潜在的な問題や改善点のリスト。"
    )
    retrieved_info: Optional[str] = Field(
        None,
        description="応答を生成するために使用された、知識ベースからの検索結果。"
    )