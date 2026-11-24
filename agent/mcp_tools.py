"""
mcp_tools.py: connects the agent to the MCP server.

What this file does:
1. Starts the MCP server (mcp_server/server.py) in the background and connects to it.
   They talk through stdio (the server's input/output), not through a port.
2. Asks the server for its tools (get_applicant, get_credit_history,
   predict_credit_risk, explain_prediction).
3. Converts each MCP tool into a LangChain tool, so the agent can use it.
   - MCP tool       = a description only (name, description, arguments);
                      the real function runs inside the MCP server.
   - LangChain tool = a Python object with the description + a function to run.
   The adapter copies the description and creates a function that forwards
   each call to the MCP server (client.call_tool).
4. Handles errors: if a tool fails (unknown applicant, API down), the error
   message is returned as text instead of crashing, so the LLM can read it.

Run `python -m agent.mcp_tools` to test the tools by hand (no LLM).
The API must be running for predict_credit_risk and explain_prediction.
"""


import asyncio
import os
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException   # NEW
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def server_params() -> StdioServerParameters:
    """How to start the MCP server."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        #env=dict(os.environ),
        env={**os.environ, "MCP_LOG_LEVEL": "WARNING"},
        cwd=str(ROOT),
    )


def to_langchain_tool(client: Client, tool) -> StructuredTool:          # NEW
    """Wrap one MCP tool as a LangChain tool."""

    async def call(**arguments):
        result = await client.call_tool(tool.name, arguments)   # ask the MCP server to run it
        text = "\n".join(part.text for part in result.content if getattr(part, "text", None))
        if result.is_error:
            raise ToolException(text)                            # e.g. "applicant not found"
        return text

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.input_schema,
        handle_tool_error=True,
    )


async def load_tools(client: Client) -> list[StructuredTool]:           # NEW
    """Get all MCP tools, already converted for LangChain."""
    listing = await client.list_tools()
    return [to_langchain_tool(client, t) for t in listing.tools]


async def demo() -> None:                                                # NEW
    async with Client(server_params()) as client:
        tools = await load_tools(client)
        print("Tools ready for the agent:", [t.name for t in tools], "\n")

        by_name = {t.name: t for t in tools}

        print("--- get_applicant(125)")
        print(await by_name["get_applicant"].ainvoke({"applicant_id": 125}), "\n")

        print("--- get_applicant(99999999)  (unknown applicant)")
        print(await by_name["get_applicant"].ainvoke({"applicant_id": 99999999}), "\n")

        print("--- explain_prediction(125)  (needs the API running)")
        print(await by_name["explain_prediction"].ainvoke({"applicant_id": 125, "top_k": 3}))


if __name__ == "__main__":
    asyncio.run(demo())