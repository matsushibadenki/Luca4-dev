# /app/meta_intelligence/self_improvement/evolution.py
#
# タイトル: 自己進化システム
# 役割: AIの能力とプロンプトを継続的に改善・進化させるためのコアロジックを担う。

from __future__ import annotations
import logging
from typing import List, Dict, Any, TYPE_CHECKING

from app.models import ProblemAnalysisResult

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# --- 循環参照を避けるための型チェック時のみのインポート ---
if TYPE_CHECKING:
    from app.meta_cognition import MetaCognitiveEngine
    from app.agents import SelfImprovementAgent, SelfCorrectionAgent
    from app.analytics import AnalyticsCollector
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class SelfEvolvingSystem:
    """
    AIの自己改善と進化を司るシステム。
    メタ認知によって発見された問題に基づき、プロンプトや能力の改善を行う。
    """
    def __init__(
        self,
        meta_cognitive_engine: "MetaCognitiveEngine",
        self_improvement_agent: "SelfImprovementAgent",
        self_correction_agent: "SelfCorrectionAgent",
        analytics_collector: "AnalyticsCollector",
    ):
        """
        コンストラクタ。
        Args:
            meta_cognitive_engine (MetaCognitiveEngine): 自己評価を行うメタ認知エンジン。
            self_improvement_agent (SelfImprovementAgent): 改善策を提案するエージェント。
            self_correction_agent (SelfCorrectionAgent): 改善策を実行（プロンプト修正など）するエージェント。
            analytics_collector (AnalyticsCollector): 分析データを収集するコレクター。
        """
        self.meta_cognitive_engine = meta_cognitive_engine
        self.self_improvement_agent = self_improvement_agent
        self.self_correction_agent = self_correction_agent
        self.analytics_collector = analytics_collector
        logger.info("SelfEvolvingSystem initialized.")

    def evolve(self, potential_problems: List[ProblemAnalysisResult]):
        """
        発見された潜在的な問題に基づいて、自己進化のプロセスを開始する。
        """
        if not potential_problems:
            return

        logger.info(f"--- Self-Evolving System: {len(potential_problems)}個の潜在的な問題に基づき、進化プロセスを開始します ---")

        for problem in potential_problems:
            logger.info(f"問題 '{problem.problem_summary}' を処理中...")

            # 1. 改善策の提案
            improvement_suggestion = self.self_improvement_agent.run(problem.model_dump())
            if not improvement_suggestion or 'suggested_improvement' not in improvement_suggestion:
                logger.warning(f"問題 '{problem.problem_summary}' に対する改善策の提案が生成されませんでした。")
                continue
                
            logger.info(f"改善提案: {improvement_suggestion['suggested_improvement']}")

            # 2. 改善策の実行（例：プロンプトの修正）
            # self_correction_agentが具体的な修正アクションを実行する
            correction_result = self.self_correction_agent.run(
                problem=problem,
                suggestion=improvement_suggestion['suggested_improvement']
            )

            # 3. 分析データ収集
            self.analytics_collector.log_event("self_evolution_cycle", {
                "problem": problem.model_dump(),
                "suggestion": improvement_suggestion,
                "correction_result": correction_result,
            })
            logger.info(f"問題 '{problem.problem_summary}' の改善サイクルが完了しました。")

        logger.info("--- Self-Evolving System: 進化プロセスが完了しました ---")