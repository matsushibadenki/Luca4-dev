# /app/resource_arbiter.py
#
# タイトル: リソースアービター
# 役割: システムリソース（主に認知エネルギー）の割り当てと消費を管理する。
#       どのパイプラインがどれだけのリソースを消費するかを決定し、エネルギーが枯渇しないように調整する。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.meta_intelligence.cognitive_energy.manager import CognitiveEnergyManager

logger = logging.getLogger(__name__)

class ResourceArbiter:
    """
    システムリソース（主に認知エネルギー）の割り当てと消費を管理する。
    """
    def __init__(self, energy_manager: "CognitiveEnergyManager"):
        """
        コンストラクタ。
        Args:
            energy_manager (CognitiveEnergyManager): 認知エネルギーを管理するマネージャー。
        """
        self.energy_manager = energy_manager
        logger.info("ResourceArbiter initialized.")

    def consume_energy(self, pipeline_name: str) -> bool:
        """
        指定されたパイプラインの実行コストに基づいて認知エネルギーを消費する。
        """
        # ここでは、パイプライン名に基づいてコストを単純にマッピングする。
        # 将来的には、より動的なコスト計算が可能。
        cost_mapping = {
            "simple": 1,
            "full": 10,
            "parallel": 15,
            "quantum": 20,
            "speculative": 5,
            "self_discover": 25,
            "internal_dialogue": 18,
            "conceptual_reasoning": 22,
            "micro_llm_expert": 8,
        }
        
        cost = cost_mapping.get(pipeline_name, 5) # デフォルトコスト
        
        logger.info(f"Pipeline '{pipeline_name}' is attempting to consume {cost} energy units.")
        
        if self.energy_manager.get_current_energy() >= cost:
            self.energy_manager.consume_energy(cost)
            logger.info(f"Energy consumed. Remaining energy: {self.energy_manager.get_current_energy()}")
            return True
        else:
            logger.warning(f"Not enough energy to run pipeline '{pipeline_name}'. Required: {cost}, Available: {self.energy_manager.get_current_energy()}")
            # エネルギーが不足している場合、回復を待つか、低コストのタスクを実行するなどの戦略が必要になる
            return False