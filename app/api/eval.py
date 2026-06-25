from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.eval_runner import EvalRunner

router = APIRouter(prefix="/api/v1/eval", tags=["eval"])
runner = EvalRunner()


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TestCaseCreate(BaseModel):
    input: str
    expected: str
    category: Optional[str] = None
    difficulty: str = "medium"


class DatasetImport(BaseModel):
    cases: list[TestCaseCreate]


class RunCreate(BaseModel):
    dataset_id: int
    provider: str
    model: str
    prompt_template: Optional[str] = None


@router.post("/datasets")
def create_dataset(body: DatasetCreate):
    dataset = runner.create_dataset(name=body.name, description=body.description)
    return {"id": dataset.id, "name": dataset.name}


@router.get("/datasets")
def list_datasets():
    datasets = runner.list_datasets()
    return [{"id": d.id, "name": d.name, "description": d.description, "created_at": d.created_at.isoformat()} for d in datasets]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    dataset = runner.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    cases = runner.get_test_cases(dataset_id)
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "cases": [{"id": c.id, "input": c.input_text, "expected": c.expected_output, "category": c.category, "difficulty": c.difficulty} for c in cases],
    }


@router.post("/datasets/{dataset_id}/cases")
def add_test_cases(dataset_id: int, body: DatasetImport):
    dataset = runner.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    cases = [c.model_dump() for c in body.cases]
    count = runner.add_test_cases(dataset_id, cases)
    return {"added": count}


@router.post("/runs")
async def create_and_run_eval(body: RunCreate):
    run = runner.create_run(
        dataset_id=body.dataset_id,
        provider_name=body.provider,
        model=body.model,
        prompt_template=body.prompt_template,
    )
    run = await runner.execute_run(run.id)
    return {
        "id": run.id,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed_cases,
        "failed": run.failed_cases,
        "pass_rate": f"{(run.passed_cases / run.total_cases * 100):.1f}%" if run.total_cases > 0 else "N/A",
        "avg_latency_ms": round(run.avg_latency_ms, 1),
        "total_tokens": run.total_tokens,
        "estimated_cost": round(run.estimated_cost, 4),
    }


@router.get("/runs")
def list_runs(dataset_id: Optional[int] = None):
    runs = runner.list_runs(dataset_id=dataset_id)
    return [
        {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "provider": r.provider,
            "model": r.model,
            "status": r.status,
            "total_cases": r.total_cases,
            "passed": r.passed_cases,
            "failed": r.failed_cases,
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "estimated_cost": round(r.estimated_cost, 4),
            "created_at": r.created_at.isoformat(),
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/runs/compare")
def compare_runs(run_a: int, run_b: int):
    run_a_obj = runner.get_run(run_a)
    run_b_obj = runner.get_run(run_b)
    if not run_a_obj or not run_b_obj:
        raise HTTPException(status_code=404, detail="Run not found")

    results_a = {r.test_case_id: r for r in runner.get_run_results(run_a)}
    results_b = {r.test_case_id: r for r in runner.get_run_results(run_b)}

    regressions = []
    improvements = []

    all_case_ids = set(results_a.keys()) | set(results_b.keys())
    for cid in all_case_ids:
        ra = results_a.get(cid)
        rb = results_b.get(cid)
        ra_passed = ra.status == "pass" if ra else False
        rb_passed = rb.status == "pass" if rb else False

        case = None
        from sqlmodel import Session
        from app.database import get_engine
        from app.models.eval import TestCase
        with Session(get_engine()) as session:
            case = session.get(TestCase, cid)

        entry = {
            "test_case_id": cid,
            "input_text": case.input_text[:100] if case else "",
            "expected_output": case.expected_output if case else "",
            "run_a_output": ra.actual_output[:80] if ra and ra.actual_output else (ra.error_message[:80] if ra and ra.error_message else ""),
            "run_a_score": round(ra.score, 2) if ra and ra.score is not None else 0,
            "run_b_output": rb.actual_output[:80] if rb and rb.actual_output else (rb.error_message[:80] if rb and rb.error_message else ""),
            "run_b_score": round(rb.score, 2) if rb and rb.score is not None else 0,
        }

        if ra_passed and not rb_passed:
            regressions.append(entry)
        elif not ra_passed and rb_passed:
            improvements.append(entry)

    avg_a = sum(r.score or 0 for r in results_a.values()) / len(results_a) if results_a else 0
    avg_b = sum(r.score or 0 for r in results_b.values()) / len(results_b) if results_b else 0
    pass_rate_a = run_a_obj.passed_cases / run_a_obj.total_cases if run_a_obj.total_cases else 0
    pass_rate_b = run_b_obj.passed_cases / run_b_obj.total_cases if run_b_obj.total_cases else 0

    return {
        "run_a": {"id": run_a_obj.id, "provider": run_a_obj.provider, "model": run_a_obj.model, "avg_score": round(avg_a, 2), "passed": run_a_obj.passed_cases, "failed": run_a_obj.failed_cases, "total_cases": run_a_obj.total_cases},
        "run_b": {"id": run_b_obj.id, "provider": run_b_obj.provider, "model": run_b_obj.model, "avg_score": round(avg_b, 2), "passed": run_b_obj.passed_cases, "failed": run_b_obj.failed_cases, "total_cases": run_b_obj.total_cases},
        "delta_avg_score": round(avg_b - avg_a, 2),
        "delta_pass_rate": round(pass_rate_b - pass_rate_a, 2),
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }


@router.get("/runs/{run_id}")
def get_run(run_id: int):
    run = runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = runner.get_run_results(run_id)
    return {
        "id": run.id,
        "provider": run.provider,
        "model": run.model,
        "status": run.status,
        "total_cases": run.total_cases,
        "passed": run.passed_cases,
        "failed": run.failed_cases,
        "avg_latency_ms": round(run.avg_latency_ms, 1),
        "total_tokens": run.total_tokens,
        "estimated_cost": round(run.estimated_cost, 4),
        "results": [
            {
                "id": r.id,
                "test_case_id": r.test_case_id,
                "status": r.status,
                "actual_output": r.actual_output,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "error_message": r.error_message,
                "failure_reason": r.failure_reason,
            }
            for r in results
        ],
    }
