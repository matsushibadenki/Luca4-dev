# /app/agents/refactoring_agent.py
# title: リファクタリングAIエージェント
# role: コードベースを分析し、品質改善のためのリファクタリング案を提案する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any, Dict

from app.agents.base import AIAgent

REFACTORING_PROMPT = ChatPromptTemplate.from_template(
    """あなたは経験豊富なソフトウェアエンジニアです。以下のコードスニペットを分析し、可読性、保守性、パフォーマンスの観点からリファクタリング案を提案してください。
    提案は、具体的な修正前後のコードと、その変更理由を明確に記述してください。

    分析対象コード ({file_path}):
    {code_snippet}

    現在のコードベースに関するコンテキスト (意味的コードグラフの要約):
    {semantic_graph_summary}
    ---
    リファクタリング提案:
    """
)


class RefactoringAgent(AIAgent):
    """
    コード品質を改善するためのリファクタリング提案を行うAIエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = REFACTORING_PROMPT
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        リファクタリングエージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("RefactoringAgent expects a dictionary as input.")

        if self._chain is None:
            raise RuntimeError("RefactoringAgent's chain is not initialized.")
        result: str = self._chain.invoke(input_data)
        return result