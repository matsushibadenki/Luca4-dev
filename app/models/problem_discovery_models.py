# /app/models/problem_discovery_models.py
#
# タイトル: 問題発見関連モデル
# 役割: 自己評価や問題発見のプロセスで使用されるデータ構造を定義する。

from pydantic import BaseModel, Field
from typing import List

class ProblemAnalysisResult(BaseModel):
    """
    自己評価プロセスによって特定された、単一の潜在的な問題に関する分析結果。
    """
    problem_summary: str = Field(
        ...,
        description="特定された問題の簡潔な要約。"
    )
    impact_area: str = Field(
        ...,
        description="問題が影響を与える可能性のある領域（例：プロンプト、知識ベース、推論プロセス）。"
    )
    severity: int = Field(
        ...,
        ge=1,
        le=5,
        description="問題の深刻度（1から5のスケール）。"
    )
    suggested_verification: str = Field(
        ...,
        description="この問題が実際に存在するかどうかを検証するための具体的な提案。"
    )

class ProblemDiscoveryOutput(BaseModel):
    """
    ProblemDiscoveryAgentからの出力。特定された問題のリストを含む。
    """
    identified_problems: List[ProblemAnalysisResult] = Field(
        default_factory=list,
        description="特定された潜在的な問題のリスト。"
    )