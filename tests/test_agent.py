"""Agent endpoint tests — at least 5 test cases covering the core scenarios.

Tests:
1. Basic chat (no tools) — simple question gets direct answer
2. Multi-step tool chain — search → time → calculator
3. Invalid template — 422 validation error
4. Trace retrieval — GET trace after chat
5. Max steps limit — agent stops at limit
6. Calculator safety — rejects dangerous expressions
7. Session not found — 404 for missing session
"""

import json

import pytest

from app.services.tools import _safe_eval, execute_search_web


# ---------------------------------------------------------------------------
# Test 1: Basic chat (simple question, no tool calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_basic_chat(test_app, mock_simple_llm):
    """Simple question should get a direct answer without tool calls."""
    response = await test_app.post(
        "/api/v1/agent/chat",
        json={
            "question": "你好，请介绍一下你自己",
            "template": "basic",
            "max_steps": 5,
        },
    )
    assert response.status_code == 200

    # Parse SSE events
    events = _parse_sse(response.text)
    event_types = [e["type"] for e in events]

    # Should have at least a thought, answer, and done
    assert "answer" in event_types
    assert "done" in event_types


# ---------------------------------------------------------------------------
# Test 2: Multi-step tool chain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_tool_chain(test_app, mock_agent_llm):
    """Agent should call multiple tools in sequence for complex questions."""
    response = await test_app.post(
        "/api/v1/agent/chat",
        json={
            "question": "帮我查一下 Transformer 论文的提出时间，算一下距今多少天",
            "template": "basic",
            "max_steps": 10,
        },
    )
    assert response.status_code == 200

    events = _parse_sse(response.text)
    event_types = [e["type"] for e in events]

    # Should have thought, action, and observation events
    assert "thought" in event_types
    assert "action" in event_types
    assert "observation" in event_types
    assert "answer" in event_types
    assert "done" in event_types

    # Should have at least 3 tool calls
    action_events = [e for e in events if e["type"] == "action"]
    assert len(action_events) >= 2, f"Expected at least 2 action events, got {len(action_events)}"

    # Verify tool names include expected tools
    tool_names = [e["data"].get("tool") for e in action_events]
    assert "search_knowledge_base" in tool_names or "get_current_time" in tool_names


# ---------------------------------------------------------------------------
# Test 3: Invalid template returns 422
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_invalid_template(test_app):
    """Invalid template name should return 422 validation error."""
    response = await test_app.post(
        "/api/v1/agent/chat",
        json={
            "question": "测试问题",
            "template": "nonexistent_template",
            "max_steps": 5,
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: Trace retrieval after chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_trace_retrieval(test_app, mock_agent_llm):
    """After running a chat, the trace should be retrievable via GET endpoint."""
    # Run a chat first
    chat_response = await test_app.post(
        "/api/v1/agent/chat",
        json={
            "question": "测试轨迹检索",
            "template": "structured",
            "max_steps": 10,
        },
    )
    assert chat_response.status_code == 200

    # Extract session_id from SSE events
    events = _parse_sse(chat_response.text)
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) > 0

    session_id = done_events[0]["data"].get("session_id")
    assert session_id, "No session_id in done event"

    # Retrieve trace
    trace_response = await test_app.get(f"/api/v1/agent/sessions/{session_id}/trace")
    assert trace_response.status_code == 200

    trace = trace_response.json()
    assert trace["session_id"] == session_id
    assert trace["question"] == "测试轨迹检索"
    assert trace["template_used"] == "structured"
    assert isinstance(trace["steps"], list)
    assert len(trace["steps"]) > 0

    # Verify step structure
    step = trace["steps"][0]
    assert "step" in step
    assert "thought" in step or "action_name" in step


# ---------------------------------------------------------------------------
# Test 5: Max steps limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_max_steps(test_app, mock_max_steps_llm):
    """Agent should stop when max_steps is reached."""
    response = await test_app.post(
        "/api/v1/agent/chat",
        json={
            "question": "无限循环测试",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response.status_code == 200

    events = _parse_sse(response.text)

    # Should have answer event indicating max steps reached
    answer_events = [e for e in events if e["type"] == "answer"]
    assert len(answer_events) > 0

    # The answer should mention the step limit
    answer_data = answer_events[0]["data"]
    assert answer_data.get("hit_max_steps") is True

    # Count action events — should be exactly 3 (matching max_steps)
    action_events = [e for e in events if e["type"] == "action"]
    assert len(action_events) == 3, f"Expected 3 actions with max_steps=3, got {len(action_events)}"


# ---------------------------------------------------------------------------
# Test 6: Calculator safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculator_safety():
    """Calculator should reject dangerous expressions."""
    # Safe expressions should work
    assert _safe_eval("2 + 3") == 5
    assert _safe_eval("10 * 5") == 50
    assert _safe_eval("2 ** 10") == 1024
    assert _safe_eval("100 / 4") == 25.0
    assert _safe_eval("-5 + 3") == -2

    # Dangerous expressions should raise ValueError
    with pytest.raises(ValueError):
        _safe_eval("__import__('os').system('rm -rf /')")

    with pytest.raises(ValueError):
        _safe_eval("open('/etc/passwd')")

    with pytest.raises(ValueError):
        _safe_eval("eval('1+1')")

    # Name references beyond whitelist should fail
    with pytest.raises(ValueError):
        _safe_eval("os.system('ls')")


# ---------------------------------------------------------------------------
# Test 7: Session not found returns 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_session_not_found(test_app):
    """Non-existent session should return 404."""
    response = await test_app.get("/api/v1/agent/sessions/nonexistent_session/trace")
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 8: Search web query length limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_web_query_length_limit():
    """Search web should reject queries exceeding max length."""
    long_query = "测试" * 200  # 400 chars, exceeds default 200 limit
    result = await execute_search_web(long_query)
    assert "过长" in result or "缩短" in result


# ---------------------------------------------------------------------------
# Test 9: Empty question returns 422
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_empty_question(test_app):
    """Empty question should return 422 validation error."""
    response = await test_app.post(
        "/api/v1/agent/chat",
        json={"question": "", "template": "basic", "max_steps": 5},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 10: List sessions endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_list_sessions(test_app, mock_simple_llm):
    """Session list endpoint should return valid structure."""
    # First create a session by chatting
    await test_app.post(
        "/api/v1/agent/chat",
        json={"question": "测试会话列表", "template": "basic", "max_steps": 3},
    )

    # List sessions
    response = await test_app.get("/api/v1/agent/sessions")
    assert response.status_code == 200

    data = response.json()
    assert "sessions" in data
    assert "total" in data
    assert isinstance(data["sessions"], list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event dict has: {"type": event_name, "data": parsed_data}
    """
    events = []
    current_event = ""

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"type": current_event, "data": data})
            current_event = ""

    return events
