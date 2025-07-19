# app/pipelines/pipeline_registry.py
#
# タイトル: パイプラインレジストリ
# 役割: システムに存在するすべての手続き記憶（ワークフロー/パイプライン）を登録・管理する。
#       中央実行系がタスクに応じて適切なパイプラインを選択する際の参照先となる。

from injector import singleton
from typing import Dict, List

@singleton
class PipelineRegistry:
    """
    利用可能なすべてのパイプラインを管理するレジストリクラス。
    """
    def __init__(self):
        """
        コンストラクタ。パイプライン情報を初期化する。
        """
        self._pipelines: Dict[str, str] = {
            "simple_pipeline": "簡単な質問や対話に対する迅速な応答を生成する基本的なワークフロー。",
            "full_pipeline": "複雑な問題に対し、計画、情報収集、自己評価を含む包括的な思考プロセスを実行するワークフロー。",
            "self_discover_pipeline": "未知の問題に直面した際に、思考モジュールを動的に組み合わせ、解決戦略を自律的に構築する高度なワークフロー。",
            "internal_dialogue_pipeline": "一つの問題に対して複数の視点から検討するため、AIエージェント間の内省的な対話を実行するワークフロー。",
            # 新しい手続き記憶（パイプライン）はここに追加していく
        }
        self._default_pipeline = "simple_pipeline"

    def get_pipeline_descriptions(self) -> Dict[str, str]:
        """
        すべての登録済みパイプラインの名前と説明の辞書を返す。

        Returns:
            Dict[str, str]: パイプライン名と説明のマッピング。
        """
        return self._pipelines

    def get_pipeline_names(self) -> List[str]:
        """
        すべての登録済みパイプラインの名前のリストを返す。

        Returns:
            List[str]: パイプライン名のリスト。
        """
        return list(self._pipelines.keys())

    def get_default_pipeline_name(self) -> str:
        """
        デフォルトのパイプライン名を返す。

        Returns:
            str: デフォルトパイプラインの名前。
        """
        return self._default_pipeline