# /app/tools/tool_belt.py
#
# タイトル: ツールベルト
# 役割: 利用可能なすべてのツール（Web検索、ブラウザ操作、専門家MicroLLMなど）を管理・提供する。

from __future__ import annotations
import logging
from typing import Dict, List, TYPE_CHECKING

from app.tools.base import Tool
from app.tools.playwright_browser_tool import PlaywrightBrowserTool
from app.tools.tavily_search_tool import TavilySearchTool
from app.tools.wikipedia_search_tool import WikipediaSearchTool

# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
# from app.micro_llm.tool import MicroLLMTool  <- この行をここから削除します
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

if TYPE_CHECKING:
    from app.llm_providers.base import LLMProvider
    from app.micro_llm.manager import MicroLLMManager

logger = logging.getLogger(__name__)

class ToolBelt:
    """
    利用可能なすべてのツールを管理するクラス。
    """
    def __init__(self, llm_provider: "LLMProvider", micro_llm_manager: "MicroLLMManager"):
        """
        コンストラクタ。
        Args:
            llm_provider (LLMProvider): LLMインスタンスを提供するためのプロバイダー。
            micro_llm_manager (MicroLLMManager): 専門家MicroLLMを管理するマネージャー。
        """
        self.llm_provider = llm_provider
        self.micro_llm_manager = micro_llm_manager
        self.tools = self._initialize_tools()
        logger.info(f"ToolBelt initialized with {len(self.tools)} tools: {list(self.tools.keys())}")

    def _initialize_tools(self) -> Dict[str, Tool]:
        """
        利用可能なすべてのツールを初期化して辞書として返す。
        """
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # 循環参照を避けるため、このメソッド内でインポートします
        from app.micro_llm.tool import MicroLLMTool
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        tools: Dict[str, Tool] = {
            "tavily_search": TavilySearchTool(),
            "wikipedia_search": WikipediaSearchTool(),
            # Playwrightは非同期なので、現時点では直接のツールとしては追加が難しい
            # "browse_web": PlaywrightBrowserTool(),
        }

        # 専門家MicroLLMをツールとして追加
        try:
            specialized_models = self.micro_llm_manager.list_specialized_models()
            for model_name in specialized_models:
                tool_name = f"micro_llm_{model_name.replace('-', '_')}"
                tools[tool_name] = MicroLLMTool(
                    llm_provider=self.llm_provider,
                    model_name=model_name
                )
        except Exception as e:
            logger.error(f"専門家MicroLLMのツールとしての初期化に失敗しました: {e}")

        return tools

    def get_tool_names(self) -> List[str]:
        """
        利用可能なすべてのツールの名前のリストを返す。
        """
        return list(self.tools.keys())

    def get_tool(self, tool_name: str) -> Tool | None:
        """
        指定された名前のツールを取得する。
        """
        return self.tools.get(tool_name)

    def get_all_tools(self) -> List[Tool]:
        """
        すべてのツールオブジェクトのリストを返す。
        """
        return list(self.tools.values())