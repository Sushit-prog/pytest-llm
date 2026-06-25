from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from datetime import datetime, timezone
import json


def _utcnow():
    return datetime.now(timezone.utc)


class EvalDataset(SQLModel, table=True):
    __tablename__ = "eval_datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    test_cases: list["TestCase"] = Relationship(back_populates="dataset")
    eval_runs: list["EvalRun"] = Relationship(back_populates="dataset")


class TestCase(SQLModel, table=True):
    __tablename__ = "test_cases"

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="eval_datasets.id")
    input_text: str
    expected_output: str
    category: Optional[str] = None
    difficulty: str = "medium"
    metadata_json: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)

    dataset: Optional[EvalDataset] = Relationship(back_populates="test_cases")

    def set_metadata(self, data: dict):
        self.metadata_json = json.dumps(data)

    def get_metadata(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}


class EvalRun(SQLModel, table=True):
    __tablename__ = "eval_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="eval_datasets.id")
    provider: str
    model: str
    prompt_template: Optional[str] = None
    status: str = "running"
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    dataset: Optional[EvalDataset] = Relationship(back_populates="eval_runs")
    results: list["EvalResult"] = Relationship(back_populates="run")


class EvalResult(SQLModel, table=True):
    __tablename__ = "eval_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="eval_runs.id")
    test_case_id: int = Field(foreign_key="test_cases.id")
    status: str  # pass, fail, error
    actual_output: Optional[str] = None
    score: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error_message: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)

    run: Optional[EvalRun] = Relationship(back_populates="results")
