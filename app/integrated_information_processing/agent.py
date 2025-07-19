# /app/integrated_information_processing/agent.py
#
# タイトル: 統合情報エージェント
# 役割: 複数の情報源や思考の断片を統合し、より高次の洞察や一貫した出力を生成する。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Any

from app.agents.base import AIAgent
from app.prompts.manager import PromptManager

class IntegratedInformationAgent(AIAgent):
    """
    複数の情報ストリームを統合し、全体的な意味や結論を導き出すエージェント。
    """
    def __init__(self, llm: Any, output_parser: Any):
        """
        コンストラクタ。
        Args:
            llm (Any): 使用する言語モデル。
            output_parser (Any): LLMの出力を整形するパーサー。
        """
        self.llm = llm
        self.output_parser = output_parser
        super().__init__()

    def build_chain(self, prompt_template: ChatPromptTemplate) -> Runnable:
        """
        実行時にプロンプトを受け取ってチェーンを構築する。
        """
        return prompt_template | self.llm | self.output_parser

    def run(self, context: str, query: str) -> str:
        """
        与えられたコンテキストとクエリを統合して、洞察を生成する。
        """
        prompt = ChatPromptTemplate.from_template(
            """
            あなたは高度な情報統合AIです。以下の複数の情報源から得られた断片的なコンテキストを分析し、
            ユーザーの質問に対して一貫性のある包括的な回答を生成してください。

            ## コンテキスト
            {context}

            ## ユーザーの質問
            {query}
            
            ## 統合された回答
            """
        )
        chain = self.build_chain(prompt)
        return chain.invoke({"context": context, "query": query})