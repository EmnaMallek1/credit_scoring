"""
__main__.py: the terminal command for the agent.

Usage:
    python -m agent "Analyse applicant 125"
    python -m agent "Is applicant 100 approved?"

Needs: GROQ_API_KEY in .env, and the API running (docker run -p 8000:8000 credit-scoring-api).
"""
import asyncio
import json
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from .credit_agent import ask


def show_tool_calls(messages: list) -> None:
    """Print only which tools the agent called (short trace)."""
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                print(f"  -> {call['name']}({json.dumps(call['args'])})")


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "Analyse applicant 125"
    print(f"Question: {question}\n\nTool calls:")
    messages = asyncio.run(ask(question))
    show_tool_calls(messages)
    print("\n" + messages[-1].content)      # the last message = the final answer


if __name__ == "__main__":
    main()