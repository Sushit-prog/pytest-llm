from fastapi import APIRouter
from sqlmodel import Session, select, func
from app.database import get_engine
from app.models.eval import EvalRun, EvalResult
from app.models.trace import Trace, ProviderUsage

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_summary():
    with Session(get_engine()) as session:
        total_runs = session.exec(select(func.count(EvalRun.id))).one()
        total_traces = session.exec(select(func.count(Trace.id))).one()
        total_results = session.exec(select(func.count(EvalResult.id))).one()
        passed = session.exec(select(func.count(EvalResult.id)).where(EvalResult.status == "pass")).one()
        failed = session.exec(select(func.count(EvalResult.id)).where(EvalResult.status == "fail")).one()

    return {
        "total_eval_runs": total_runs,
        "total_traces": total_traces,
        "total_eval_results": total_results,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{(passed / total_results * 100):.1f}%" if total_results > 0 else "N/A",
    }


@router.get("/providers")
def get_provider_stats():
    with Session(get_engine()) as session:
        runs = list(session.exec(select(EvalRun)).all())

    stats = {}
    for r in runs:
        key = f"{r.provider}/{r.model}"
        if key not in stats:
            stats[key] = {"provider": r.provider, "model": r.model, "calls": 0, "total_tokens": 0, "total_cost": 0.0, "total_latency": 0.0, "total_score": 0.0, "total_cases": 0}
        stats[key]["calls"] += 1
        stats[key]["total_tokens"] += r.total_tokens
        stats[key]["total_cost"] += r.estimated_cost
        stats[key]["total_latency"] += r.avg_latency_ms * r.total_cases
        stats[key]["total_cases"] += r.total_cases
        stats[key]["total_score"] += r.passed_cases

    return [
        {
            "provider": v["provider"],
            "model": v["model"],
            "provider_model": k,
            "calls": v["calls"],
            "total_tokens": v["total_tokens"],
            "total_cost": round(v["total_cost"], 4),
            "avg_latency_ms": round(v["total_latency"] / v["total_cases"], 1) if v["total_cases"] > 0 else 0,
            "pass_rate": round(v["total_score"] / v["total_cases"] * 100, 1) if v["total_cases"] > 0 else 0,
        }
        for k, v in stats.items()
    ]


@router.get("/failures")
def get_failure_summary():
    with Session(get_engine()) as session:
        results = list(
            session.exec(select(EvalResult).where(EvalResult.status.in_(["fail", "error"]))).all()
        )

    failure_types = {}
    for r in results:
        key = r.error_message if r.error_message else "wrong_output"
        if key not in failure_types:
            failure_types[key] = 0
        failure_types[key] += 1

    return [{"type": k, "count": v} for k, v in sorted(failure_types.items(), key=lambda x: -x[1])[:10]]


@router.get("/recommendations")
def get_recommendations():
    with Session(get_engine()) as session:
        runs = list(session.exec(select(EvalRun).where(EvalRun.status == "completed")).all())

    if not runs:
        return {"best_overall": None, "best_accuracy": None, "fastest": None, "cheapest": None}

    stats = {}
    for r in runs:
        key = f"{r.provider}/{r.model}"
        if key not in stats:
            stats[key] = {"provider": r.provider, "model": r.model, "pass_rates": [], "latencies": [], "costs": []}
        if r.total_cases > 0:
            stats[key]["pass_rates"].append(r.passed_cases / r.total_cases * 100)
            stats[key]["latencies"].append(r.avg_latency_ms)
            stats[key]["costs"].append(r.estimated_cost)

    providers = []
    for key, v in stats.items():
        avg_pass = sum(v["pass_rates"]) / len(v["pass_rates"]) if v["pass_rates"] else 0
        avg_lat = sum(v["latencies"]) / len(v["latencies"]) if v["latencies"] else 99999
        avg_cost = sum(v["costs"]) / len(v["costs"]) if v["costs"] else 99999
        providers.append({"key": key, "provider": v["provider"], "model": v["model"], "pass_rate": avg_pass, "latency": avg_lat, "cost": avg_cost})

    if not providers:
        return {"best_overall": None, "best_accuracy": None, "fastest": None, "cheapest": None}

    # Normalize for composite score
    max_pass = max(p["pass_rate"] for p in providers) or 1
    min_lat = min(p["latency"] for p in providers) or 1
    max_lat = max(p["latency"] for p in providers) or 1
    min_cost = min(p["cost"] for p in providers) or 1
    max_cost = max(p["cost"] for p in providers) or 1

    for p in providers:
        lat_norm = 1 - ((p["latency"] - min_lat) / (max_lat - min_lat)) if max_lat > min_lat else 1
        cost_norm = 1 - ((p["cost"] - min_cost) / (max_cost - min_cost)) if max_cost > min_cost else 1
        p["composite"] = (p["pass_rate"] / 100 * 0.6) + (lat_norm * 0.25) + (cost_norm * 0.15)

    best_overall = max(providers, key=lambda p: p["composite"])
    best_accuracy = max(providers, key=lambda p: p["pass_rate"])
    fastest = min(providers, key=lambda p: p["latency"])
    cheapest = min(providers, key=lambda p: p["cost"])

    def fmt(p):
        return {"provider": p["provider"], "model": p["model"], "key": p["key"]}

    return {
        "best_overall": {**fmt(best_overall), "score": round(best_overall["composite"], 3), "reason": f"Composite score {best_overall['composite']:.3f} (accuracy {best_overall['pass_rate']:.0f}%, {best_overall['latency']:.0f}ms)"},
        "best_accuracy": {**fmt(best_accuracy), "score": round(best_accuracy["pass_rate"], 1), "reason": f"Highest pass rate at {best_accuracy['pass_rate']:.1f}%"},
        "fastest": {**fmt(fastest), "score": round(fastest["latency"], 1), "reason": f"Lowest latency at {fastest['latency']:.0f}ms"},
        "cheapest": {**fmt(cheapest), "score": round(cheapest["cost"], 4), "reason": f"Lowest cost at ${cheapest['cost']:.4f}"},
    }
