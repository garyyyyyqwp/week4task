"""Session management — in-memory store with JSON file persistence.

Stores agent reasoning traces (AgentResult) keyed by session_id.
Each session is saved to disk as a JSON file for durability.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.config import AGENT_SESSION_DIR

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """A single step in the ReAct reasoning trace."""

    step_number: int
    thought: str | None = None
    action_name: str | None = None
    action_input: dict | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    """Complete result of an agent run."""

    session_id: str
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    template_used: str = "basic"
    created_at: str = ""


# In-memory session store
_sessions: dict[str, AgentResult] = {}
_SAVE_DIR = Path(AGENT_SESSION_DIR)
_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _result_to_dict(result: AgentResult) -> dict[str, Any]:
    """Convert AgentResult to a JSON-serializable dict."""
    return {
        "session_id": result.session_id,
        "question": result.question,
        "steps": [
            {
                "step_number": s.step_number,
                "thought": s.thought,
                "action_name": s.action_name,
                "action_input": s.action_input,
                "observation": s.observation,
            }
            for s in result.steps
        ],
        "final_answer": result.final_answer,
        "total_steps": result.total_steps,
        "template_used": result.template_used,
        "created_at": result.created_at,
    }


def _dict_to_result(data: dict[str, Any]) -> AgentResult:
    """Reconstruct AgentResult from a dict."""
    steps = [
        AgentStep(
            step_number=s["step_number"],
            thought=s.get("thought"),
            action_name=s.get("action_name"),
            action_input=s.get("action_input"),
            observation=s.get("observation"),
        )
        for s in data.get("steps", [])
    ]
    return AgentResult(
        session_id=data["session_id"],
        question=data["question"],
        steps=steps,
        final_answer=data.get("final_answer", ""),
        total_steps=data.get("total_steps", 0),
        template_used=data.get("template_used", "basic"),
        created_at=data.get("created_at", ""),
    )


def save_session(result: AgentResult) -> None:
    """Save agent result to memory and disk.

    Args:
        result: The AgentResult to persist.
    """
    _sessions[result.session_id] = result
    path = _SAVE_DIR / f"{result.session_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_result_to_dict(result), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save session %s: %s", result.session_id, e)


def get_session(session_id: str) -> AgentResult | None:
    """Get session by ID, loading from disk if not in memory.

    Args:
        session_id: The session identifier.

    Returns:
        AgentResult if found, None otherwise.
    """
    if session_id in _sessions:
        return _sessions[session_id]

    path = _SAVE_DIR / f"{session_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = _dict_to_result(data)
            _sessions[session_id] = result
            return result
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    return None


def list_sessions() -> list[dict[str, Any]]:
    """List all sessions with summary info.

    Returns:
        List of session summary dicts, newest first.
    """
    results: list[dict[str, Any]] = []

    # Check in-memory sessions first
    for session_id, result in _sessions.items():
        results.append({
            "session_id": result.session_id,
            "question": result.question[:60],
            "total_steps": result.total_steps,
            "template": result.template_used,
            "created_at": result.created_at,
        })

    # Check disk for sessions not in memory
    seen_ids = {r["session_id"] for r in results}
    for path in sorted(_SAVE_DIR.glob("*.json"), reverse=True):
        session_id = path.stem
        if session_id in seen_ids:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "session_id": data["session_id"],
                "question": data.get("question", "")[:60],
                "total_steps": data.get("total_steps", 0),
                "template": data.get("template_used", "basic"),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            continue

    # Sort by created_at descending
    results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return results


def clear_sessions() -> None:
    """Clear all sessions from memory and disk. For testing only."""
    _sessions.clear()
    for path in _SAVE_DIR.glob("*.json"):
        try:
            path.unlink()
        except Exception:
            pass
