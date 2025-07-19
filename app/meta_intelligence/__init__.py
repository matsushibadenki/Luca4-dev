# /app/meta_intelligence/__init__.py
#
# タイトル: メタインテリジェンス パッケージ初期化
# 役割: このパッケージ内のすべてのコンポーネントを外部に公開する。

from .cognitive_energy.manager import CognitiveEnergyManager
from .self_improvement.evolution import SelfEvolvingSystem
# emergent と value_evolution のディレクトリ構造を仮定
from .emergent.network import EmergentIntelligenceNetwork
from .value_evolution.values import EvolvingValueSystem
from .evolutionary_controller import EvolutionaryController

__all__ = [
    "CognitiveEnergyManager",
    "SelfEvolvingSystem",
    "EmergentIntelligenceNetwork",
    "EvolvingValueSystem",
    "EvolutionaryController",
]