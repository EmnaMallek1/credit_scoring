"""
test_agent.py: tests of the agent with a fake LLM (no Groq key, no internet).

The fake LLM replays prepared answers (tool calls, then a final text).
We test our code around the LLM: the MCP server is started, the tools are given
to the LLM, tool results and errors are sent back to it.
"""
import asyncio

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import build_applicant_db
from agent.credit_agent import ask
from test_mcp_server import make_csv_frame


class ScriptedLLM(BaseChatModel):
    """A fake LLM: returns the messages of `script` one after the other."""
    script: list
    step: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self                       # accepts the tools, ignores them

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[self.step]  # next prepared answer
        self.step += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name, args, call_id):
    """Build a tool request, the way a real LLM would write it."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A small temporary database + an API address where nothing runs."""
    path = tmp_path / "applicants.db"
    build_applicant_db.write_db(make_csv_frame(), path)
    monkeypatch.setenv("APPLICANTS_DB", str(path))
    monkeypatch.setenv("CREDIT_API_URL", "http://127.0.0.1:9")
    return path


def run(llm, question="Analyse applicant 5"):
    return asyncio.run(ask(question, llm=llm))


def test_agent_reads_applicant_through_mcp(db):
    llm = ScriptedLLM(script=[
        AIMessage(content="", tool_calls=[call("get_applicant", {"applicant_id": 5}, "c1")]),
        AIMessage(content="Applicant 5 profile retrieved."),
    ])
    messages = run(llm)
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "success"
    assert '"applicant_id": 5' in tool_msg.content
    assert messages[-1].content == "Applicant 5 profile retrieved."


def test_unknown_applicant_error_reaches_the_llm(db):
    llm = ScriptedLLM(script=[
        AIMessage(content="", tool_calls=[call("get_applicant", {"applicant_id": 999999}, "c1")]),
        AIMessage(content="Applicant not found."),
    ])
    tool_msg = next(m for m in run(llm) if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "not found" in tool_msg.content


def test_api_down_error_reaches_the_llm(db):
    llm = ScriptedLLM(script=[
        AIMessage(content="", tool_calls=[call("predict_credit_risk", {"applicant_id": 5}, "c1")]),
        AIMessage(content="The scoring API is unavailable."),
    ])
    tool_msg = next(m for m in run(llm) if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "Docker container" in tool_msg.content


def test_all_four_mcp_tools_are_given_to_the_llm(db):
    seen = {}

    class Recorder(ScriptedLLM):
        def bind_tools(self, tools, **kwargs):
            seen["names"] = {t.name for t in tools}   # remember which tools it received
            return self

    run(Recorder(script=[AIMessage(content="ok")]))
    assert seen["names"] == {"get_applicant", "get_credit_history",
                             "predict_credit_risk", "explain_prediction"}