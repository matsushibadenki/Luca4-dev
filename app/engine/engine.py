# /app/engine/engine.py
# title: メタインテリジェンス・エンジン
# role: ユーザーからの入力を受け付け、リソースを割り当て、最適な推論パイプラインを実行する司令塔。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# from app.containers import container # この行を完全に削除します
from app.models import OrchestrationDecision
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

if TYPE_CHECKING:
    from app.pipelines.base import BasePipeline
    from app.resource_arbiter import ResourceArbiter
    from app.models import MasterAgentResponse

logger = logging.getLogger(__name__)

class MetaIntelligenceEngine:
    """
    推論パイプラインの選択と実行を管理するコアエンジン。
    """
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def __init__(self, pipelines: Dict[str, "BasePipeline"], resource_arbiter: "ResourceArbiter"):
        """
        コンストラクタ。
        Args:
            pipelines (Dict[str, "BasePipeline"]): DIコンテナから注入される、利用可能なすべてのパイプラインの辞書。
            resource_arbiter (ResourceArbiter): リソース（認知エネルギー）を管理するアービター。
        """
        self.pipelines = pipelines
        self.resource_arbiter = resource_arbiter
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def run(self, orchestration_decision: OrchestrationDecision, query: str) -> "MasterAgentResponse":
        """
        オーケストレーションの決定に基づき、適切なパイプラインを実行する。
        """
        pipeline_name = orchestration_decision.get("selected_pipeline", "simple")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        pipeline = self.pipelines.get(pipeline_name)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        if not pipeline:
            logger.error(f"要求されたパイプライン '{pipeline_name}' が見つかりません。")
            # デフォルトのパイプラインにフォールバックするなどの処理を追加できる
            default_pipeline_name = "simple"
            pipeline = self.pipelines.get(default_pipeline_name)
            if not pipeline:
                 raise ValueError("デフォルトのパイプライン 'simple' さえも見つかりません。")
            logger.warning(f"デフォルトのパイプライン '{default_pipeline_name}' にフォールバックします。")


        logger.info(f"--- MetaIntelligence Engine: '{pipeline_name}' パイプラインを実行します ---")
        
        # リソース割り当て（エネルギー消費）
        self.resource_arbiter.consume_energy(pipeline_name)
        
        # パイプラインの実行
        result = pipeline.run(query=query, orchestration_decision=orchestration_decision)
        
        logger.info(f"--- MetaIntelligence Engine: '{pipeline_name}' パイプラインの実行が完了しました ---")
        return result