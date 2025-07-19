# app/agents/central_executive_agent.py
#
# タイトル: 中央実行系エージェント
# 役割: ユーザーからの高レベルな目標を理解し、それを達成するための最適なワークフロー（パイプライン）を決定する司令塔。
#       タスクの複雑さ、種類、および利用可能な手続き記憶を評価し、最も効果的な処理経路を選択する。

from injector import inject, singleton
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.pipelines.pipeline_registry import PipelineRegistry

@singleton
class CentralExecutiveAgent:
    """
    ユーザーの要求を分析し、最適なパイプラインを選択する中央実行系。
    """

    # プロンプトテンプレート：利用可能なパイプラインのリストとユーザーの要求を基に、最適なものを選択させる
    PROMPT_TEMPLATE = """
あなたは、高度なAI開発アシスタントシステム「Luca」の中央実行系です。
あなたの役割は、ユーザーからの要求を分析し、それを処理するための最も適切な内部ワークフロー（パイプライン）を選択することです。

利用可能なワークフローは以下の通りです。
---
{available_pipelines}
---

上記のワークフローの説明を参考に、以下のユーザー要求を処理するために最も適したワークフローの「名前」を一つだけ選択してください。
あなたの応答は、選択したワークフローの正確な名前のみでなければなりません。余計な説明は含めないでください。

ユーザー要求:
{user_query}

最適なワークフロー名:
"""

    @inject
    def __init__(self, llm: LLM, pipeline_registry: PipelineRegistry):
        """
        コンストラクタ。DIコンテナによりLLMとPipelineRegistryが注入される。

        Args:
            llm (LLM): 言語モデル。
            pipeline_registry (PipelineRegistry): 利用可能なパイプラインを管理するレジストリ。
        """
        self.llm = llm
        self.pipeline_registry = pipeline_registry
        self.prompt = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def select_pipeline(self, query: str) -> str:
        """
        与えられたクエリに最適なパイプラインを選択する。

        Args:
            query (str): ユーザーからのクエリ。

        Returns:
            str: 選択されたパイプラインの名前。
        """
        available_pipelines = self.pipeline_registry.get_pipeline_descriptions()
        
        # 利用可能なパイプラインの情報を整形
        formatted_pipelines = "\n".join(
            [f"- {name}: {description}" for name, description in available_pipelines.items()]
        )

        # チェーンを実行して最適なパイプライン名を取得
        selected_pipeline_name = self.chain.invoke({
            "available_pipelines": formatted_pipelines,
            "user_query": query
        })
        
        # LLMの出力が正確な名前であることを保証する
        if selected_pipeline_name in self.pipeline_registry.get_pipeline_names():
            return selected_pipeline_name
        else:
            # マッチしない場合は、デフォルトのパイプラインを返すなどのフォールバック処理
            # ここではシンプルに最初のパイプラインを返す
            return self.pipeline_registry.get_default_pipeline_name()