"""
credit_agent.py: the agent itself.

LLM (Groq) + MCP tools, assembled with create_agent (LangChain).
create_agent builds a LangGraph loop: the LLM reads the question, asks for
tools, receives their results, and repeats until it writes the final answer.
"""
import asyncio
import os


from langchain.agents import create_agent
from langchain_groq import ChatGroq
from mcp import Client

from .mcp_tools import load_tools, server_params

from .prompts import SYSTEM_PROMPT

MAX_STEPS = 15   # safety limit, so the agent can't loop forever


def make_llm() -> ChatGroq:
    """The LLM: model name read from .env (GROQ_MODEL)."""
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)


async def ask(question: str, llm=None) -> list:
    """Run the agent on one question; return all messages of the conversation.
    llm: the LLM to use. None = the real Groq LLM (tests pass a fake one)."""
    async with Client(server_params()) as client:
        tools = await load_tools(client)
        agent = create_agent(llm or make_llm(), tools, system_prompt=SYSTEM_PROMPT)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": MAX_STEPS},
        )
    return result["messages"]




