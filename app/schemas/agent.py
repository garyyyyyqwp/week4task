"""Pydantic v2 schemas for Agent API endpoints."""

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Agent Chat
# ---------------------------------------------------------------------------

class AgentChatRequest(BaseModel):
    """Request for agent chat endpoint."""

    question: str = Field(
        ..., min_length=1, max_length=5000,
        description="用户问题",
    )
    template: str = Field(
        default="basic",
        description="ReAct 模板: basic | structured | self_correcting",
    )
    max_steps: int = Field(
        default=10, ge=1, le=20,
        description="最大推理步数",
    )
    image_url: str | None = Field(
        default=None,
        description="可选图片URL（用于多模态场景，图片直达主模型推理）",
    )
    image_base64: str | None = Field(
        default=None,
        description="可选图片的base64编码（data:image/...;base64,...格式）。"
                    "与image_url二选一，图片直接进入多模态主模型，零信息损失。",
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate that the template name is one of the allowed options."""
        valid = {"basic", "structured", "self_correcting"}
        if v not in valid:
            raise ValueError(
                f"无效的 ReAct 模板: '{v}'。可选: {', '.join(sorted(valid))}"
            )
        return v


# ---------------------------------------------------------------------------
# Trace / Session
# ---------------------------------------------------------------------------

class StepInfo(BaseModel):
    """A single step in the agent reasoning trace."""

    step: int = Field(..., description="步骤编号")
    thought: str | None = Field(default=None, description="Agent的推理思考")
    action_name: str | None = Field(default=None, description="调用的工具名称")
    action_input: dict | None = Field(default=None, description="工具输入参数")
    observation: str | None = Field(default=None, description="工具返回结果")


class TraceResponse(BaseModel):
    """Full reasoning trace for an agent session."""

    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")
    steps: list[StepInfo] = Field(default_factory=list, description="推理步骤列表")
    final_answer: str = Field(default="", description="最终回答")
    total_steps: int = Field(default=0, description="总步骤数")
    template_used: str = Field(default="basic", description="使用的ReAct模板")
    created_at: str = Field(default="", description="创建时间(ISO格式)")


class SessionInfo(BaseModel):
    """Summary info for a session in the list view."""

    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题(截断)")
    total_steps: int = Field(default=0, description="总步骤数")
    template: str = Field(default="basic", description="使用的ReAct模板")
    created_at: str = Field(default="", description="创建时间(ISO格式)")


class SessionListResponse(BaseModel):
    """Response for session list endpoint."""

    sessions: list[SessionInfo] = Field(default_factory=list, description="会话列表")
    total: int = Field(default=0, description="会话总数")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationTestCase(BaseModel):
    """A single test case for agent evaluation."""

    question: str = Field(..., min_length=1, description="测试问题")
    expected_answer: str | None = Field(default=None, description="期望答案(可选)")


class EvaluateRequest(BaseModel):
    """Request for agent evaluation endpoint."""

    test_cases: list[EvaluationTestCase] = Field(
        ..., min_length=1, max_length=10,
        description="测试用例列表(1-10个)",
    )
    templates: list[str] = Field(
        default=["basic", "structured", "self_correcting"],
        description="要对比的ReAct模板列表",
    )
    max_steps: int = Field(default=10, ge=1, le=20, description="最大推理步数")

    @field_validator("templates")
    @classmethod
    def validate_templates(cls, v: list[str]) -> list[str]:
        """Validate that all template names are valid."""
        valid = {"basic", "structured", "self_correcting"}
        for t in v:
            if t not in valid:
                raise ValueError(f"无效的 ReAct 模板: '{t}'。可选: {', '.join(sorted(valid))}")
        return v


class TemplateEvaluationResult(BaseModel):
    """Evaluation result for a single template."""

    template: str = Field(..., description="模板名称")
    avg_task_completion: float = Field(default=0.0, description="平均任务完成度")
    avg_tool_accuracy: float = Field(default=0.0, description="平均工具选择准确度")
    avg_hallucination_score: float = Field(default=0.0, description="平均信息准确性")
    avg_reasoning_quality: float = Field(default=0.0, description="平均推理质量")
    avg_total_score: float = Field(default=0.0, description="加权总分")
    avg_steps: float = Field(default=0.0, description="平均步骤数")
    runs: list[dict] = Field(default_factory=list, description="各测试用例的运行详情")


class EvaluateResponse(BaseModel):
    """Response for agent evaluation endpoint."""

    results: list[TemplateEvaluationResult] = Field(
        default_factory=list, description="各模板的评估结果",
    )
    recommendation: str = Field(
        default="", description="推荐使用的模板及理由",
    )
