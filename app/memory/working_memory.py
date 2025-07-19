# app/memory/working_memory.py
#
# タイトル: ワーキングメモリ
# 役割: 現在のタスクに直接関連する情報を一時的に保持する領域。
#       長期記憶からロードされた情報、ユーザーとの対話履歴、AIの中間的な思考などを格納し、
#       LLMがタスクを処理するためのコンテキストを形成する。

from injector import singleton
from typing import List, Dict, Any, Optional

@singleton
class WorkingMemory:
    """
    タスク実行に必要な情報を一時的に保持するワーキングメモリ。
    人間の脳のワーキングメモリのように、現在アクティブなタスクに関連する情報を保持する。
    """

    def __init__(self):
        """
        コンストラクタ。ワーキングメモリを初期化する。
        """
        self.current_task: Optional[str] = None
        self.task_context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def load_task(self, task_description: str, context: Dict[str, Any]):
        """
        新しいタスクをワーキングメモリにロードする。

        Args:
            task_description (str): 実行するタスクの説明。
            context (Dict[str, Any]): 長期記憶などから取得したタスク関連コンテキスト。
        """
        self.current_task = task_description
        self.task_context = context
        # 新しいタスクを開始する際に、短期的な対話履歴はクリアしても良いかもしれない
        self.conversation_history = []

    def update_context(self, key: str, value: Any):
        """
        現在のタスクコンテキストを更新する。

        Args:
            key (str): コンテキストのキー。
            value (Any): 更新する値。
        """
        self.task_context[key] = value

    def add_to_history(self, role: str, content: str):
        """
        対話履歴を追加する。

        Args:
            role (str): 発話者（例: "user", "assistant"）。
            content (str): 発話内容。
        """
        self.conversation_history.append({"role": role, "content": content})

    def get_full_context(self) -> Dict[str, Any]:
        """
        現在のタスクに関するすべての情報を結合して返す。

        Returns:
            Dict[str, Any]: タスク説明、コンテキスト、対話履歴を含む辞書。
        """
        return {
            "task": self.current_task,
            "context": self.task_context,
            "history": self.conversation_history
        }

    def clear(self):
        """
        ワーキングメモリをクリアする。
        """
        self.current_task = None
        self.task_context = {}
        self.conversation_history = []