# /app/agents/knowledge_graph_agent.py
#
# タイトル: 知識グラフ生成AIエージェント
# 役割: テキストから知識グラフを構築する。

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent
from app.knowledge_graph.models import KnowledgeGraph as KnowledgeGraphModel
from app.prompts.manager import PromptManager

class KnowledgeGraphAgent(AIAgent):
    """
    テキストから知識グラフを生成するAIエージェント。
    """
    def __init__(self, llm: Any, prompt_manager: PromptManager):
        self.llm = llm
        # プロンプトマネージャーから必要なプロンプトを取得
        self.prompt_template = prompt_manager.get_prompt("KNOWLEDGE_GRAPH_AGENT_PROMPT")
        self.output_parser = JsonOutputParser(pydantic_object=KnowledgeGraphModel)
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        知識グラフ生成エージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> KnowledgeGraphModel:
        """
        テキストから知識グラフを生成して返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("KnowledgeGraphAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("KnowledgeGraphAgent's chain is not initialized.")
        
        result_from_chain = self._chain.invoke(input_data)

        if isinstance(result_from_chain, dict):
            return KnowledgeGraphModel.model_validate(result_from_chain)
        elif isinstance(result_from_chain, KnowledgeGraphModel):
            return result_from_chain
        else:
            raise TypeError(f"KnowledgeGraphAgentのチェーンから予期せぬ型が返されました: {type(result_from_chain)}")