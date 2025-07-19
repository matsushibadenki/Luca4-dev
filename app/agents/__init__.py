# /app/agents/__init__.py
#
# タイトル: エージェントパッケージ初期化
# 役割: /app/agents/ ディレクトリ内に直接存在するすべてのAIエージェントをインポートし、
#       __all__リストを通じて外部からアクセス可能にする。

from .base import AIAgent
from .planning_agent import PlanningAgent
from .self_correction_agent import SelfCorrectionAgent
from .knowledge_graph_agent import KnowledgeGraphAgent
from .tool_using_agent import ToolUsingAgent
from .retrieval_evaluator_agent import RetrievalEvaluatorAgent
from .query_refinement_agent import QueryRefinementAgent
from .thinking_modules import DecomposeAgent, CritiqueAgent, SynthesizeAgent
from .self_improvement_agent import SelfImprovementAgent
from .autonomous_agent import AutonomousAgent
from .consolidation_agent import ConsolidationAgent
from .knowledge_gap_analyzer import KnowledgeGapAnalyzerAgent
from .capability_mapper_agent import CapabilityMapperAgent
from .orchestration_agent import OrchestrationAgent
from .master_agent import MasterAgent
from .performance_benchmark_agent import PerformanceBenchmarkAgent
from .cognitive_loop_agent import CognitiveLoopAgent
from .complexity_analyzer import ComplexityAnalyzer

# 注: WorldModelAgent, SelfCriticAgent, DialogueParticipantAgent などは
# app/cognitive_modeling/ や app/meta_cognition/ のような別のパッケージに属するため、
# このファイルではインポートしません。DIコンテナが直接それらの場所からインポートします。

__all__ = [
    "AIAgent",
    "PlanningAgent",
    "SelfCorrectionAgent",
    "KnowledgeGraphAgent",
    "ToolUsingAgent",
    "RetrievalEvaluatorAgent",
    "QueryRefinementAgent",
    "DecomposeAgent",
    "CritiqueAgent",
    "SynthesizeAgent",
    "SelfImprovementAgent",
    "AutonomousAgent",
    "ConsolidationAgent",
    "KnowledgeGapAnalyzerAgent",
    "CapabilityMapperAgent",
    "OrchestrationAgent",
    "MasterAgent",
    "PerformanceBenchmarkAgent",
    "CognitiveLoopAgent",
    "ComplexityAnalyzer",
]