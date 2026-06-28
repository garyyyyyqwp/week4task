"""Agent Router — ReAct agent chat with SSE streaming + trace retrieval."""

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.schemas.agent import (
    AgentChatRequest,
    TraceResponse,
    StepInfo,
    SessionInfo,
    SessionListResponse,
    EvaluateRequest,
    EvaluateResponse,
    TemplateEvaluationResult,
)
from app.services.agent import run_agent_stream, run_agent_sync
from app.services.sessions import get_session, list_sessions
from app.services.prompts import REACT_PROMPT_TEMPLATES
from app.services.evaluator import evaluate_agent_run

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agent"])


# ---------------------------------------------------------------------------
# POST /api/v1/agent/chat — SSE streaming agent chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def agent_chat(request: AgentChatRequest):
    """Agent chat with SSE streaming.

    Each reasoning step (Thought/Action/Observation) is sent as an SSE event.
    The final answer is sent as the 'answer' event, followed by 'done'.
    """
    session_id = uuid.uuid4().hex[:12]

    async def event_generator():
        async for event in run_agent_stream(
            question=request.question,
            session_id=session_id,
            template=request.template,
            max_steps=request.max_steps,
            image_url=request.image_url,
        ):
            yield event

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# GET /api/v1/agent/sessions — List all sessions
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=SessionListResponse)
async def list_all_sessions():
    """List all agent sessions with summary info."""
    sessions = list_sessions()
    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions],
        total=len(sessions),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/agent/sessions/{session_id}/trace — Full reasoning trace
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/trace",
    response_model=TraceResponse,
    responses={404: {"description": "会话不存在"}},
)
async def get_trace(session_id: str):
    """Return full reasoning trace as JSON."""
    result = get_session(session_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话不存在: {session_id}",
        )

    return TraceResponse(
        session_id=result.session_id,
        question=result.question,
        steps=[
            StepInfo(
                step=s.step_number,
                thought=s.thought,
                action_name=s.action_name,
                action_input=s.action_input,
                observation=s.observation,
            )
            for s in result.steps
        ],
        final_answer=result.final_answer,
        total_steps=result.total_steps,
        template_used=result.template_used,
        created_at=result.created_at,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/agent/evaluate — Compare ReAct templates
# ---------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_templates(request: EvaluateRequest):
    """Run agent evaluation across multiple ReAct prompt templates.

    For each template × test case combination, runs the agent and evaluates
    the result using LLM-as-Judge.
    """
    results: list[TemplateEvaluationResult] = []

    for template in request.templates:
        runs = []

        for tc in request.test_cases:
            session_id = uuid.uuid4().hex[:12]
            try:
                agent_result = await run_agent_sync(
                    question=tc.question,
                    session_id=session_id,
                    template=template,
                    max_steps=request.max_steps,
                )

                # Evaluate the run
                verdict = await evaluate_agent_run(
                    question=tc.question,
                    steps=[
                        {
                            "thought": s.thought,
                            "action_name": s.action_name,
                            "action_input": s.action_input,
                            "observation": s.observation,
                        }
                        for s in agent_result.steps
                    ],
                    final_answer=agent_result.final_answer,
                    expected_answer=tc.expected_answer,
                )

                run_detail = {
                    "question": tc.question,
                    "session_id": session_id,
                    "total_steps": agent_result.total_steps,
                    "final_answer": agent_result.final_answer[:200],
                    "verdict": verdict.model_dump() if verdict else None,
                }

            except Exception as e:
                logger.error("Evaluation run failed for template=%s: %s", template, e)
                run_detail = {
                    "question": tc.question,
                    "session_id": session_id,
                    "error": str(e),
                }

            runs.append(run_detail)

        # Compute averages for this template
        valid_verdicts = [r["verdict"] for r in runs if r.get("verdict")]
        valid_steps = [r.get("total_steps", 0) for r in runs if "total_steps" in r]

        if valid_verdicts:
            avg_tc = sum(v["task_completion"] for v in valid_verdicts) / len(valid_verdicts)
            avg_ta = sum(v["tool_selection_accuracy"] for v in valid_verdicts) / len(valid_verdicts)
            avg_h = sum(v["hallucination_rate"] for v in valid_verdicts) / len(valid_verdicts)
            avg_rq = sum(v["reasoning_quality"] for v in valid_verdicts) / len(valid_verdicts)
            avg_total = sum(
                v["task_completion"] * 0.35
                + v["tool_selection_accuracy"] * 0.20
                + v["hallucination_rate"] * 0.25
                + v["reasoning_quality"] * 0.20
                for v in valid_verdicts
            ) / len(valid_verdicts)
        else:
            avg_tc = avg_ta = avg_h = avg_rq = avg_total = 0.0

        avg_steps = sum(valid_steps) / len(valid_steps) if valid_steps else 0.0

        results.append(TemplateEvaluationResult(
            template=template,
            avg_task_completion=round(avg_tc, 2),
            avg_tool_accuracy=round(avg_ta, 2),
            avg_hallucination_score=round(avg_h, 2),
            avg_reasoning_quality=round(avg_rq, 2),
            avg_total_score=round(avg_total, 2),
            avg_steps=round(avg_steps, 1),
            runs=runs,
        ))

    # Generate recommendation
    if results:
        best = max(results, key=lambda r: r.avg_total_score)
        recommendation = (
            f"推荐使用「{best.template}」模板，"
            f"加权总分最高（{best.avg_total_score:.2f}/5.00），"
            f"平均{best.avg_steps:.1f}步完成推理。"
        )
    else:
        recommendation = "无评估结果。"

    return EvaluateResponse(
        results=results,
        recommendation=recommendation,
    )
