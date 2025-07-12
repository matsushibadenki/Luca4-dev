# /app/tools/program_construction_tool.py
# title: プログラム構築ツール
# role: AI_sandboxのProgramConstructionAgentをLangChainツールとしてラップし、Lucaの他のエージェントが利用できるようにする。

import logging
from typing import Any, Dict, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# AI_sandboxから必要なものをインポート (パスを修正)
from app.tools.program_construction_agent import ProgramConstructionAgent
from app.sandbox.sandbox_manager.service import SandboxManagerService # これもDIで渡されるはず
from app.tools.sandbox_tools import SandboxTool # SandboxToolはProgramConstructionAgentの依存性なので、これもDIで渡す

logger = logging.getLogger(__name__)

class ProgramConstructionToolInput(BaseModel):
    """ProgramConstructionToolの入力スキーマ"""
    user_requirement: str = Field(description="The user's high-level requirement for program construction or code modification.")
    llm_agent_id: str = Field(description="The unique ID of the LLM agent (Luca's internal agent ID) requesting the program construction. This is used by the sandbox to maintain persistence.")

class ProgramConstructionTool(BaseTool):
    """
    AI_sandboxのProgramConstructionAgentをラップしたLangChainツール。
    LucaのAIが自身のコードを生成・実行・デバッグしたり、機能拡張を行ったりするために使用する。
    """
    name: str = "ProgramConstructionTool"
    description: str = (
        "This tool allows the AI to autonomously construct, execute, and debug code (Python/Node.js) "
        "within an isolated and persistent Docker sandbox. "
        "It can be used for self-extension, creating new functionalities, "
        "modifying existing code, or executing arbitrary scripts to achieve complex tasks. "
        "Provide a detailed `user_requirement` outlining the desired program or action. "
        "Always include the `llm_agent_id` for persistent sandbox sessions."
    )
    args_schema: Type[BaseModel] = ProgramConstructionToolInput

    # DIでProgramConstructionAgentそのものを注入する
    program_construction_agent: ProgramConstructionAgent

    def _run(self, user_requirement: str, llm_agent_id: str, run_manager: Optional[Any] = None) -> str:
        logger.info(f"ProgramConstructionTool: Received request from agent {llm_agent_id} to construct program for: {user_requirement}")
        try:
            result = self.program_construction_agent.run_program_construction(user_requirement, llm_agent_id)
            logger.info(f"ProgramConstructionTool: Program construction completed with result: {result}")
            return result
        except Exception as e:
            logger.error(f"ProgramConstructionTool: Error during program construction: {e}", exc_info=True)
            return f"Error executing program construction: {str(e)}"

    async def _arun(self, user_requirement: str, llm_agent_id: str, run_manager: Optional[Any] = None) -> str:
        # ProgramConstructionAgent.run_program_construction は内部でasyncio.runを使用しているため、
        # ここでは同期的に呼び出しても問題ない (または、ProgramConstructionAgentにも_arunを実装する)。
        # 現状はrun_program_constructionが同期的に呼び出してもOKなので_runを呼び出す
        return self._run(user_requirement=user_requirement, llm_agent_id=llm_agent_id, run_manager=run_manager)