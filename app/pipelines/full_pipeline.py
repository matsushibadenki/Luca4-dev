# /app/pipelines/full_pipeline.py
# title: フル思考パイプライン
# role: 複雑なタスクに対して、計画、認知ループ、自己進化を含む包括的な思考プロセスを実行する。

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING
import asyncio

from app.models import MasterAgentResponse, OrchestrationDecision
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from .base import BasePipeline
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

if TYPE_CHECKING:
    from app.agents import (
        MasterAgent, PlanningAgent, CognitiveLoopAgent, ProblemDiscoveryAgent
    )
    from app.memory import MemoryConsolidator
    from app.analytics import AnalyticsCollector
    from app.meta_intelligence.self_improvement.evolution import SelfEvolvingSystem

logger = logging.getLogger(__name__)

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
class FullPipeline(BasePipeline):
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    """
    複雑なタスクに対応するための、包括的で自己進化可能な思考パイプライン。
    """
    def __init__(
        self,
        master_agent: "MasterAgent",
        planning_agent: "PlanningAgent",
        cognitive_loop_agent: "CognitiveLoopAgent",
        meta_cognitive_engine: "MetaCognitiveEngine",
        problem_discovery_agent: "ProblemDiscoveryAgent",
        memory_consolidator: "MemoryConsolidator",
        analytics_collector: "AnalyticsCollector",
        self_evolving_system: "SelfEvolvingSystem",
    ):
        self.master_agent = master_agent
        self.planning_agent = planning_agent
        self.cognitive_loop_agent = cognitive_loop_agent
        self.meta_cognitive_engine = meta_cognitive_engine
        self.problem_discovery_agent = problem_discovery_agent
        self.memory_consolidator = memory_consolidator
        self.analytics_collector = analytics_collector
        self.self_evolving_system = self_evolving_system

    async def arun(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        start_time = time.time()
        logger.info("--- Full Pipeline START ---")

        # 1. Plan
        plan = self.planning_agent.build_chain().invoke({"query": query})
        
        # 2. Execute Cognitive Loop
        cognitive_loop_output = self.cognitive_loop_agent.run(query, plan)
        
        # 3. Metacognitive Reflection
        self_criticism, potential_problems = self.meta_cognitive_engine.run(query, cognitive_loop_output)
        
        # 4. Generate Final Answer
        final_answer = self.master_agent.run(
            query=query,
            plan=plan,
            retrieved_info=cognitive_loop_output,
            self_criticism=self_criticism
        )

        # 5. Consolidate Memory
        self.memory_consolidator.consolidate_interaction(
            query=query,
            plan=plan,
            response=final_answer,
            retrieved_info=cognitive_loop_output,
            criticism=self_criticism,
            problems=potential_problems
        )

        # 6. Evolve System (if necessary)
        if potential_problems:
             self.self_evolving_system.evolve(potential_problems)

        end_time = time.time()
        logger.info(f"--- Full Pipeline END ({(end_time - start_time):.2f} s) ---")

        return {
            "final_answer": final_answer,
            "self_criticism": self_criticism,
            "potential_problems": str(potential_problems), # Pydantic互換のため文字列に
            "retrieved_info": cognitive_loop_output
        }

    def run(self, query: str, orchestration_decision: OrchestrationDecision) -> MasterAgentResponse:
        """同期版のrunメソッド"""
        return asyncio.run(self.arun(query, orchestration_decision))