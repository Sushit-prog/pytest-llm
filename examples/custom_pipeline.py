"""
Example: Using the AI Reliability Platform tracer SDK

This demonstrates how to instrument your own LLM pipelines
with automatic tracing and span collection.

Run with: python examples/custom_pipeline.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tracer import trace_llm_call
from app.services.providers.registry import get_provider


async def classify_email(email_text: str) -> str:
    """Simple email classifier using Groq."""
    provider = get_provider("groq")
    response = await provider.complete(
        prompt=email_text,
        model="llama-3.3-70b-versatile",
        system_prompt="Classify this email as spam, important, or promotional. Reply with only the label.",
    )
    return response.content.strip()


async def run_pipeline():
    """Run a multi-step pipeline with tracing."""
    test_emails = [
        "Subject: You won a FREE iPhone! Click here now!",
        "Subject: Q3 Budget Review Meeting - Action Required",
        "Subject: 50% off all shoes this weekend only!",
    ]

    for email in test_emails:
        print(f"\nProcessing: {email[:50]}...")

        async with trace_llm_call(name="email_classifier", provider="groq", model="llama-3.3-70b-versatile") as span:
            span.set_input(email)

            result = await classify_email(email)

            span.set_output(result)
            span.set_tokens(50, 10)

            print(f"  Result: {result}")

    print("\nDone! Check the traces page at http://localhost:8000/traces")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
