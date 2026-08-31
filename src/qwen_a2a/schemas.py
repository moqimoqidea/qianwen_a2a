"""Qwen-specific HTTP callback models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QwenEventCallback(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    answer_id: str | None = None
    event_type: Literal["audit.failed", "conversation.terminated"]
    event_data: dict[str, object] = Field(default_factory=dict)


class QwenCallbackResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: dict[str, object] = Field(default_factory=dict)
