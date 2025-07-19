# /app/agents/complexity_analyzer.py
#
# タイトル: 複雑性分析エージェント
# 役割: ユーザーのクエリやタスクの複雑性を評価し、最適な処理パイプラインの選択を支援する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from typing import Any, Dict

from app.agents.base import AIAgent

class ComplexityAnalyzer(AIAgent):
    """
    ユーザーのクエリの複雑性を分析し、構造化されたデータとして出力するエージェント。
    """
    def __init__(self, llm: Any):
        """
        コンストラクタ。
        Args:
            llm (Any): 使用する言語モデル。
        """
        self.llm = llm
        # 複雑性分析に特化したプロンプト
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            あなたはタスク分析の専門家です。以下のユーザー要求を分析し、その複雑性を評価してください。
            評価基準は以下の通りです。
            - 単純さ (1-5): 1が最も単純、5が最も複雑。
            - 曖昧さ (1-5): 1が最も明確、5が最も曖昧。
            - 必要な知識の広さ (1-5): 1が限定的、5が広範。

            ユーザー要求: "{query}"

            評価結果をJSON形式で出力してください。例: {{"simplicity": 2, "ambiguity": 1, "knowledge_scope": 3}}
            """
        )
        self.output_parser = JsonOutputParser()
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        複雑性分析用のLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        クエリの複雑性分析を実行します。
        """
        if self._chain is None:
            raise RuntimeError("ComplexityAnalyzer's chain is not initialized.")
        return self._chain.invoke({"query": query})