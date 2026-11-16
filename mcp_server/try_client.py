"""Manual test: start the MCP server like an AI app would, list its tools and call them.

    python -m mcp_server.try_client            # uses applicant 125
    python -m mcp_server.try_client 4021       # another applicant

The credit scoring API must be running (docker run -p 8000:8000 credit-scoring-api).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main(applicant_id: int) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=dict(os.environ),
        cwd=str(ROOT),
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        print(f"{len(tools.tools)} tools exposed:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")

        for name, args in [("get_applicant", {"applicant_id": applicant_id}),
                           ("get_credit_history", {"applicant_id": applicant_id}),
                           ("predict_credit_risk", {"applicant_id": applicant_id}),
                           ("explain_prediction", {"applicant_id": applicant_id, "top_k": 3})]:
            result = await client.call_tool(name, args)
            text = result.content[0].text
            print(f"\n== {name}{'  (ERROR)' if result.is_error else ''}")
            try:
                print(json.dumps(json.loads(text), indent=2))
            except json.JSONDecodeError:
                print(text)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 125))