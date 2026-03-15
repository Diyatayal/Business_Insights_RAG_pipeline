import streamlit as st
import json
from langchain_openai import ChatOpenAI
from config.config import DATA_DIR, OPENAI_API_KEY, OPENAI_MODEL
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from utils.rag_utils import get_or_create_vectorstore, retrieve_context
from utils.web_search import get_web_search_tool

import os



llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    max_tokens=1024
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    with st.spinner("Loading documents..."):
        st.session_state.vectorstore = get_or_create_vectorstore()


@tool
def search_annual_reports(query: str) -> str:
    """Search company annual reports for financials, revenue, strategy,
    growth, EBITDA, segments, balance sheet, cash flow, and any
    company-specific information. Use this FIRST for company questions."""
    return retrieve_context(query, st.session_state.vectorstore)


@tool
def search_web(query: str) -> str:
    """Search the web for general financial concepts, companies not in
    the reports, real-time news, or when annual report search has no results."""
    web_tool = get_web_search_tool()
    return str(web_tool.invoke(query))


TOOLS_LIST = [search_annual_reports, search_web]
TOOLS_MAP = {t.name: t for t in TOOLS_LIST}


llm_with_tools = llm.bind_tools(TOOLS_LIST)



def chat(user_query: str, mode: str) -> str:
    # sanitize special characters
    user_query = user_query.replace("&", "and").replace("<", "").replace(">", "")

    mode_instruction = (
        "Be CONCISE, answer in 2-4 sentences."
        if "Concise" in mode
        else "Be DETAILED with sections, bullet points, and numbers."
    )

    system = SystemMessage(content=f"""You are an expert Financial Analyst for annual report analysis.
{mode_instruction}

Use `search_annual_reports` first for any company financial question.
Use `search_web` for general concepts or if reports have no relevant data.
Always cite your source.""")


    history = []
    for msg in st.session_state.chat_history[-10:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    messages = [system] + history + [HumanMessage(content=user_query)]


    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        st.warning("⚠️ Tool calling failed, retrying without tools...")
        response = llm.invoke(messages)
        return response.content


    if not response.tool_calls:
        return response.content


    messages.append(response)

    for tc in response.tool_calls:
        tool_fn = TOOLS_MAP[tc["name"]]


        tool_label = "Searching Annual Reports..." if tc["name"] == "search_annual_reports" else "🌐 Searching the Web..."
        st.info(f"**Tool called:** `{tc['name']}` — {tool_label}")

        result = tool_fn.invoke({"query": tc["args"]["query"]})

        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"]
        ))


    final = llm.invoke(messages)
    return final.content


st.set_page_config(page_title="📊 Annual Report Analyst", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("Settings")
    response_mode = st.radio("Response Mode", ["Concise", "Detailed"])

    st.divider()
    st.markdown("### Loaded Documents")
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".pdf"):
                st.markdown(f"- `{f}`")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.title("📊 Annual Report Intelligence Chatbot")
st.caption("Ask questions about company financials, strategy, and competitive analysis.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about revenue, growth, competitors, strategy..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = chat(user_input, response_mode)
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})