from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from app.database import init_db, get_engine
from app.models.eval import EvalDataset, EvalRun, EvalResult, TestCase
from app.models.trace import Trace, Span
from app.api import eval as eval_api
from app.api import trace as trace_api
from app.api import dashboard as dashboard_api
from app.services.providers.registry import list_providers

app = FastAPI(title="AI Reliability Platform", version="0.1.0")

templates = Jinja2Templates(directory="app/templates")

app.include_router(eval_api.router)
app.include_router(trace_api.router)
app.include_router(dashboard_api.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def dashboard(request: Request):
    with Session(get_engine()) as session:
        total_runs = session.exec(select(EvalRun)).all()
        total_traces = session.exec(select(Trace)).all()
        recent_runs = sorted(total_runs, key=lambda r: r.created_at, reverse=True)[:5]
        recent_traces = sorted(total_traces, key=lambda t: t.created_at, reverse=True)[:5]
        total_results = len(session.exec(select(EvalResult)).all())
        passed = len(session.exec(select(EvalResult).where(EvalResult.status == "pass")).all())
        total_cost = sum(r.estimated_cost for r in total_runs)

    pass_rate = f"{(passed / total_results * 100):.1f}%" if total_results > 0 else "N/A"

    return templates.TemplateResponse(request, "dashboard.html", {
        "total_runs": len(total_runs),
        "total_traces": len(total_traces),
        "total_results": total_results,
        "passed": passed,
        "pass_rate": pass_rate,
        "total_cost": total_cost,
        "recent_runs": [{"id": r.id, "provider": r.provider, "model": r.model, "status": r.status, "passed_cases": r.passed_cases, "total_cases": r.total_cases, "avg_latency_ms": r.avg_latency_ms, "estimated_cost": r.estimated_cost, "created_at": r.created_at.strftime("%Y-%m-%d %H:%M")} for r in recent_runs],
        "recent_traces": recent_traces,
    })


@app.get("/eval/datasets")
def eval_datasets_page(request: Request):
    with Session(get_engine()) as session:
        datasets = list(session.exec(select(EvalDataset)).all())
    return templates.TemplateResponse(request, "eval_datasets.html", {"datasets": datasets})


@app.get("/eval/datasets/{dataset_id}")
def eval_dataset_detail_page(request: Request, dataset_id: int):
    with Session(get_engine()) as session:
        dataset = session.get(EvalDataset, dataset_id)
        cases = list(session.exec(select(TestCase).where(TestCase.dataset_id == dataset_id)).all())
    if not dataset:
        return templates.TemplateResponse(request, "404.html", {}, status_code=404)
    return templates.TemplateResponse(request, "eval_detail.html", {"dataset": dataset, "cases": cases})


@app.get("/eval/runs")
def eval_runs_page(request: Request):
    with Session(get_engine()) as session:
        runs = list(session.exec(select(EvalRun).order_by(EvalRun.created_at.desc())).all())
    return templates.TemplateResponse(request, "eval_runs.html", {"runs": runs})


@app.get("/eval/runs/compare")
def eval_compare_page(request: Request):
    with Session(get_engine()) as session:
        runs = list(session.exec(select(EvalRun).order_by(EvalRun.created_at.desc())).all())
    return templates.TemplateResponse(request, "eval_compare.html", {"runs": runs})


@app.get("/providers")
def providers_page(request: Request):
    return templates.TemplateResponse(request, "providers.html", {})


@app.get("/failures")
def failures_page(request: Request):
    return templates.TemplateResponse(request, "failure_analytics.html", {})


@app.get("/eval/runs/{run_id}")
def eval_run_detail_page(request: Request, run_id: int):
    with Session(get_engine()) as session:
        run = session.get(EvalRun, run_id)
        results = list(session.exec(select(EvalResult).where(EvalResult.run_id == run_id)).all())
        if run:
            cases = {c.id: c for c in session.exec(select(TestCase).where(TestCase.dataset_id == run.dataset_id)).all()}
        else:
            cases = {}
    if not run:
        return templates.TemplateResponse(request, "404.html", {}, status_code=404)
    return templates.TemplateResponse(request, "eval_run_detail.html", {
        "run": run, "results": results, "cases": cases,
    })


@app.get("/traces")
def traces_page(request: Request):
    with Session(get_engine()) as session:
        traces = list(session.exec(select(Trace).order_by(Trace.created_at.desc())).all())
    return templates.TemplateResponse(request, "traces.html", {"traces": traces})


@app.get("/traces/{trace_id}")
def trace_detail_page(request: Request, trace_id: str):
    with Session(get_engine()) as session:
        trace = session.exec(select(Trace).where(Trace.trace_id == trace_id)).first()
        spans = list(session.exec(select(Span).where(Span.trace_id == trace_id).order_by(Span.created_at)).all()) if trace else []
    if not trace:
        return templates.TemplateResponse(request, "404.html", {}, status_code=404)
    return templates.TemplateResponse(request, "trace_detail.html", {
        "trace": trace, "spans": spans,
    })


@app.get("/api/v1/providers")
def get_providers():
    return list_providers()
