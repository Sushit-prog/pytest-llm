"""
AI Reliability Platform — Trace SDK

Usage:
    from app.services.tracer import trace_llm_call

    async with trace_llm_call(name="my_pipeline_step") as span:
        response = await my_llm_call()
        span.set_output(response)
"""
import uuid
import time
from contextlib import asynccontextmanager
from sqlmodel import Session
from app.database import get_engine
from app.models.trace import Trace, Span


class SpanContext:
    def __init__(self, trace_id: str, span_id: str, name: str, provider: str = None, model: str = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.provider = provider
        self.model = model
        self.input_text = None
        self.output_text = None
        self.status = "success"
        self.tokens_in = 0
        self.tokens_out = 0
        self.error_message = None
        self._start = time.perf_counter()

    def set_input(self, text: str):
        self.input_text = text

    def set_output(self, text: str):
        self.output_text = text

    def set_status(self, status: str):
        self.status = status

    def set_tokens(self, tokens_in: int, tokens_out: int):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out

    def set_error(self, message: str):
        self.status = "error"
        self.error_message = message

    def _save(self):
        latency_ms = (time.perf_counter() - self._start) * 1000
        with Session(get_engine()) as session:
            span = Span(
                trace_id=self.trace_id,
                span_id=self.span_id,
                name=self.name,
                provider=self.provider,
                model=self.model,
                input_text=self.input_text,
                output_text=self.output_text,
                status=self.status,
                latency_ms=latency_ms,
                tokens_in=self.tokens_in,
                tokens_out=self.tokens_out,
                error_message=self.error_message,
            )
            session.add(span)
            session.commit()


@asynccontextmanager
async def trace_llm_call(name: str, provider: str = None, model: str = None):
    trace_id = str(uuid.uuid4())
    span_id = str(uuid.uuid4())

    trace = Trace(trace_id=trace_id, name=name)
    with Session(get_engine()) as session:
        session.add(trace)
        session.commit()

    span_ctx = SpanContext(trace_id, span_id, name, provider, model)

    try:
        yield span_ctx
    except Exception as e:
        span_ctx.set_error(str(e))
        raise
    finally:
        span_ctx._save()
        latency_ms = (time.perf_counter() - span_ctx._start) * 1000
        with Session(get_engine()) as session:
            from sqlmodel import select
            t = session.exec(select(Trace).where(Trace.trace_id == trace_id)).first()
            if t:
                t.total_latency_ms = latency_ms
                t.total_tokens = span_ctx.tokens_in + span_ctx.tokens_out
                t.status = span_ctx.status
                session.add(t)
                session.commit()
