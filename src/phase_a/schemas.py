"""Validated records written by the Phase A pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRecord(StrictModel):
    run_id: str
    commit_sha: str
    experiment_mode: Literal["exact_configuration_attempt", "quantized_reproduction", "mock"]
    model_profile: str
    split: Literal["health", "smoke", "reproduction"]
    pattern_id: str
    behavior_id: str
    sample_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**63 - 1)
    raw_user_prompt: str
    rendered_prompt_hash: str
    response_text: str
    finish_reason: str | None = None
    prompt_token_count: int | None = Field(default=None, ge=0)
    completion_token_count: int | None = Field(default=None, ge=0)
    model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    repository_revision: str
    chat_template_hash: str
    quantization_metadata: dict[str, Any]
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_tokens: int = Field(gt=0)
    reasoning_setting: bool
    generation_timestamp: datetime
    generation_duration: float = Field(ge=0)
    validity_status: bool
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("generation_timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class JudgmentPayload(StrictModel):
    """Strict JSON contract emitted by the judge model."""

    label: bool
    evidence: list[str]
    reason: str = Field(min_length=1)
    invalid: bool

    @field_validator("evidence")
    @classmethod
    def short_nonempty_evidence(cls, spans: list[str]) -> list[str]:
        if any(not span.strip() or len(span) > 300 for span in spans):
            raise ValueError("evidence spans must be nonempty and at most 300 characters")
        return spans


class JudgmentRecord(JudgmentPayload):
    run_id: str
    split: Literal["smoke", "reproduction"]
    pattern_id: str
    behavior_id: str
    sample_index: int = Field(ge=0)
    response_checksum: str
    judge_model: str
    judge_revision: str
    judged_at: datetime
    parse_attempts: int = Field(ge=1, le=2)
    raw_judge_output: str


class AuditLabel(StrictModel):
    run_id: str
    split: Literal["smoke", "reproduction"]
    pattern_id: str
    behavior_id: str
    sample_index: int = Field(ge=0)
    human_label: bool | None = None
    ambiguity_flag: bool | None = None
    notes: str = ""


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: datetime
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_remote_url: str
    git_dirty: bool
    experiment_mode: Literal["exact_configuration_attempt", "quantized_reproduction", "mock"]
    model_profile: str
    model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    dataset: str
    dataset_revision: str
    judge_model: str
    judge_revision: str
    hardware: dict[str, Any]
    software: dict[str, Any]
    quantization: dict[str, Any]
    generation_parameters: dict[str, Any]
    hashes: dict[str, str]
    seed_derivation: str
    offline: bool

    @model_validator(mode="after")
    def production_is_clean(self) -> RunManifest:
        if self.experiment_mode != "mock" and self.git_dirty:
            raise ValueError("production run manifest cannot record a dirty Git tree")
        if self.experiment_mode != "mock" and not self.git_remote_url:
            raise ValueError("production run manifest requires the Git remote URL")
        return self
