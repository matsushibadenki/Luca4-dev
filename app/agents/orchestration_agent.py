# /app/agents/orchestration_agent.py
#
# タイトル: オーケストレーションAIエージェント
# 役割: ユーザーの要求の複雑性や性質を判断し、最適な推論パイプラインを選択する司令塔。

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.models.orchestration_decision import OrchestrationDecision
from app.config import settings # 修正：settingsをインポート

if TYPE_CHECKING:
    from app.llm_providers.base import LLMProvider
    from app.agents.complexity_analyzer import ComplexityAnalyzer
    from app.tools.tool_belt import ToolBelt
    from app.prompts.manager import PromptManager

logger = logging.getLogger(__name__)

class OrchestrationAgent:
    """
    タスクに最適なパイプラインを選択し、処理フローを決定するオーケストレーター。
    """
    def __init__(
        self,
        llm_provider: "LLMProvider",
        output_parser: "JsonOutputParser",
        prompt_manager: "PromptManager",
        complexity_analyzer: "ComplexityAnalyzer",
        tool_belt: "ToolBelt",
    ):
        """
        コンストラクタ。
        """
        self.llm_provider = llm_provider
        self.output_parser = output_parser
        self.prompt_manager = prompt_manager
        self.complexity_analyzer = complexity_analyzer
        self.tool_belt = tool_belt
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 存在しないget_model_nameを呼び出す代わりに、settingsから直接モデル名を取得する
        self.llm = self.llm_provider.get_llm_instance(model=settings.GENERATION_LLM_SETTINGS["model"])
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.prompt_template = self.prompt_manager.get_prompt("ORCHESTRATION_AGENT_PROMPT")
        if not self.prompt_template:
            raise ValueError("ORCHESTRATION_AGENT_PROMPTが見つかりません。")
        self.chain = self.prompt_template | self.llm | self.output_parser

    async def determine_pipeline(self, query: str) -> Dict[str, Any]:
        """
        ユーザーのクエリに基づいて、最適なパイプラインを非同期で決定する。
        """
        logger.info(f"オーケストレーション開始: クエリ='{query}'")
        
        # 1. 複雑性の分析
        complexity_result = self.complexity_analyzer.invoke(query)
        
        # 2. 利用可能なツールのリストを取得
        available_tools = self.tool_belt.get_tool_names()

        # 3. LLMにパイプライン選択を依頼
        response = await self.chain.ainvoke({
            "query": query,
            "complexity": complexity_result,
            "available_tools": ", ".join(available_tools)
        })
        
        # Pydanticモデルで検証
        decision = OrchestrationDecision.model_validate(response)
        logger.info(f"オーケストレーション決定: {decision.model_dump_json(indent=2, ensure_ascii=False)}")
        
        return decision.model_dump()