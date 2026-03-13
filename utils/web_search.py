from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from config.config import SERP_API_KEY
import os

os.environ["SERPAPI_API_KEY"] = SERP_API_KEY

def get_web_search_tool() -> Tool:
    """Return a LangChain Tool wrapping SerpAPI."""
    try:
        search = SerpAPIWrapper()
        tool = Tool(
            name="web_search",
            func=search.run,
            description=(
                "Use this tool when the question is about general knowledge, "
                "recent events, or topics NOT found in the annual reports or documents. "
                "Input should be a clear search query."
            )
        )
        return tool
    except Exception as e:
        raise RuntimeError(f"Failed to initialize web search: {e}")