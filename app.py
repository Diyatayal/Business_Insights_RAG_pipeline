import streamlit as st
import json
from groq import Groq
from langchain_core.tools import tool

from utils.rag_utils import get_or_create_vectorstore, retrieve_context
from utils.web_search import get_web_search_tool
from config.config import DATA_DIR, GROQ_API_KEY
import os

client = Groq(api_key=GROQ_API_KEY)


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

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_reports",
            "description": "Search company annual reports for financials, revenue, strategy, growth, EBITDA, segments, balance sheet, cash flow, and any company-specific information. Use this FIRST for company questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for annual reports"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for general financial concepts, companies not in the reports, real-time news, or when annual report search has no results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for the web"}
                },
                "required": ["query"]
            }
        }
    }
]
TOOLS_MAP = {t.name: t for t in TOOLS_LIST}


def chat(user_query: str, mode: str) -> str:
    mode_instruction = (
        "Be CONCISE, answer in 2-4 sentences."
        if "Concise" in mode
        else "Be DETAILED with sections, bullet points, and numbers."
    )

    messages = [
        {
            "role": "system",
            "content": f"""You are an expert Financial Analyst for annual report analysis.
{mode_instruction}

Use `search_annual_reports` first for any company financial question.
Use `search_web` for general concepts or if reports have no relevant data.
Always cite your source."""
        }
    ]


    messages += st.session_state.chat_history[-10:]
    messages.append({"role": "user", "content": user_query})


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        max_tokens=1024
    )

    msg = response.choices[0].message


    if not msg.tool_calls:
        return msg.content


    messages.append(msg)

    for tc in msg.tool_calls:
        tool_fn = TOOLS_MAP[tc.function.name]
        args = json.loads(tc.function.arguments)
        tool_label = "📂 Searching Annual Reports..." if tc.function.name == "search_annual_reports" else "🌐 Searching the Web..."
        st.info(f"**Tool called:** `{tc.function.name}` — {tool_label}")
        result = tool_fn.invoke({"query": args["query"]})

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result)
        })


    final = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024
    )

    return final.choices[0].message.content


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