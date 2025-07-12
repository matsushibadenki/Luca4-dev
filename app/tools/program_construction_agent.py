# /app/tools/program_construction_agent.py
# title: プログラム構築エージェント
# role: LLMとツールを用いてコードの構築、実行、デバッグをサンドボックス内で行う。

from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import Tool, BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from typing import List, Dict, Any, Optional

# Lucaのパス構造に合わせてインポートを修正
from .sandbox_tools import (
    SandboxTool, ReadFileTool, WriteFileTool, ListInstalledPackagesTool,
    ListProcessesTool, CheckSyntaxTool, DiagnoseSandboxExecutionTool,
    ListDiskSpaceTool, DownloadFileTool, UploadFileTool,
    DownloadWebpageRecursivelyTool, FindFilesInSandboxTool, GrepFileContentInSandboxTool, GetSystemInfoTool
)
from .program_construction_output_parser import CustomAgentOutputParser
from app.config import settings as config # Lucaのconfigを使用
from app.sandbox.sandbox_manager.service import SandboxManagerService # Lucaのsandboxパッケージからインポート
from langchain_core.language_models.chat_models import BaseChatForToolCalling # 型ヒント用に修正

class ProgramConstructionAgent:
    def __init__(
        self,
        llm: BaseChatForToolCalling, # LLMインスタンスを直接受け取る
        sandbox_manager_service: SandboxManagerService, # SandboxManagerServiceもDIで受け取る
    ):
        self.llm = llm
        self.sandbox_manager_service = sandbox_manager_service

        # ツールリストをここで初期化し、sandbox_manager_serviceを注入
        self.tools: List[BaseTool] = [
            SandboxTool(sandbox_manager_service=self.sandbox_manager_service),
            ReadFileTool(sandbox_manager_service=self.sandbox_manager_service),
            WriteFileTool(sandbox_manager_service=self.sandbox_manager_service),
            ListInstalledPackagesTool(sandbox_manager_service=self.sandbox_manager_service),
            ListProcessesTool(sandbox_manager_service=self.sandbox_manager_service),
            CheckSyntaxTool(sandbox_manager_service=self.sandbox_manager_service),
            DiagnoseSandboxExecutionTool(sandbox_manager_service=self.sandbox_manager_service),
            ListDiskSpaceTool(sandbox_manager_service=self.sandbox_manager_service),
            DownloadFileTool(sandbox_manager_service=self.sandbox_manager_service),
            UploadFileTool(sandbox_manager_service=self.sandbox_manager_service),
            DownloadWebpageRecursivelyTool(sandbox_manager_service=self.sandbox_manager_service),
            FindFilesInSandboxTool(sandbox_manager_service=self.sandbox_manager_service),
            GrepFileContentInSandboxTool(sandbox_manager_service=self.sandbox_manager_service),
            GetSystemInfoTool(sandbox_manager_service=self.sandbox_manager_service),
        ]

        # プロンプトテンプレートの定義 (ReActフレームワークに基づき再構成)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are an autonomous program construction agent. Your primary goal is to fulfill the user's request by thinking step-by-step and using the available tools. "
             "Follow the ReAct (Reason+Act) framework: Thought, Action, Action Input, Observation.\n\n"
             "**INSTRUCTIONS:**\n"
             "1. **Analyze the Request**: Carefully understand the user's goal.\n"
             "2. **Think Step-by-Step (Thought)**: Before taking any action, you MUST use the 'Thought' field to explain your reasoning. Your thought process should include:\n"
             "   - Your understanding of the user's goal.\n"
             "   - A plan to achieve the goal, broken down into small, manageable steps.\n"
             "   - The specific tool you will use for the *next* step and why.\n"
             "3. **Choose an Action**: Select ONE tool from the available tools list: {tool_names}.\n"
             "4. **Provide Action Input**: Provide the required arguments for the chosen tool in a valid JSON format.\n"
             "5. **Wait for Observation**: After your action, you will receive an 'Observation' with the result of the tool's execution. Use this observation to inform your next thought and action.\n"
             "6. **Iterate**: Repeat the Thought-Action-Observation cycle until the user's request is fully completed.\n"
             "7. **Final Answer**: Once the goal is achieved, you MUST use the 'Final Answer:' prefix to provide the final result or code to the user. Do not include any 'Thought' or 'Action' after the 'Final Answer:'.\n\n"
             "**IMPORTANT RULES:**\n"
             "- **One Action at a Time**: You can only perform one action at a time.\n"
             "- **Use `llm_agent_id`**: Always include the `llm_agent_id`: `{llm_agent_id}` in your `Action Input`. This ID ensures your sandbox session is persistent.\n"
             "- **Use Shared Directory**: All file operations are relative to the shared directory: `{shared_dir_path}`. Use this path when executing scripts (e.g., `python {shared_dir_path}/my_script.py`).\n"
             "- **Conversational Replies**: If the user's request is a simple greeting or a question that doesn't require tools, respond directly in a conversational manner using the 'Final Answer:' prefix (e.g., 'Final Answer: Hello! How can I help you today?').\n\n"
             "**AVAILABLE TOOLS:**\n{tools}\n"
             ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            HumanMessage(content="{input}"),
        ])

        # RunnableAgent の構築
        self.agent_runnable = (
            self.prompt_template.partial(
                tools=self.tools,
                tool_names=", ".join([t.name for t in self.tools]),
                shared_dir_path=config.SHARED_DIR_CONTAINER_PATH,
            )
            | self.llm.bind_tools(self.tools) # LLMにツールをバインド
            | CustomAgentOutputParser()
        )

        self.agent = AgentExecutor(
            agent=self.agent_runnable,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True, # パースエラーをハンドルして再試行させる
            max_iterations=20 # エージェントの無限ループを防ぐための反復回数制限を少し増やす
        )

    def run_program_construction(self, user_requirement: str, llm_agent_id: str) -> str:
        # AgentExecutorは同期的に動作するため、asyncio.runを内部で呼び出す
        # LucaのメインループがFastAPI/asyncioベースであるため、ここでawaitableにするのは困難。
        # 代わりに、ProgramConstructionAgent自体が同期的なインターフェースを提供し、
        # 必要に応じて内部でasyncio.runを使用する。
        import asyncio
        print(f"ProgramConstructionAgent: Starting program construction for requirement: {user_requirement}")
        try:
            # invokeがasyncメソッドの場合
            if hasattr(self.agent, 'ainvoke'):
                result = asyncio.run(self.agent.ainvoke(
                    {
                        "input": user_requirement,
                        "llm_agent_id": llm_agent_id,
                        "agent_scratchpad": [],
                    }
                ))
            else: # invokeが同期メソッドの場合
                result = self.agent.invoke(
                    {
                        "input": user_requirement,
                        "llm_agent_id": llm_agent_id,
                        "agent_scratchpad": [],
                    }
                )
            
            output_content = result.get('output', 'No output received.') if isinstance(result, dict) else result
            print(f"ProgramConstructionAgent: Program construction finished. Result: {output_content}")
            return str(output_content)
        except Exception as e:
            print(f"ProgramConstructionAgent: An unhandled error occurred in AgentExecutor: {e}")
            return f"I'm sorry, an internal error occurred during program construction. Please try your request again. Details: {e}"