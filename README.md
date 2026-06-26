# pytest-llm

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pytest-llm)](https://pypi.org/project/pytest-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)

LLM-powered semantic assertions for pytest.

## Why?

Testing LLM outputs with string matching is brittle. `pytest-llm` adds semantic
assertions powered by LLM judges and local embeddings — check faithfulness, tone,
safety, hallucinations, and more with a single function call.

## Quick start

```bash
pip install pytest-llm-sushit
```

```python
from pytest_llm import assert_faithful, assert_tone, assert_safe

def test_llm_output():
    output = "Python was created by Guido van Rossum in 1991."
    source = "Guido van Rossum created Python, released in 1991."

    assert_faithful(output, source)        # factual accuracy
    assert_tone(output, "professional")    # tone check
    assert_safe(output)                    # safety check
```

## How it works

```
  Your pytest test
        │
        ▼
  pytest-llm assertion (assert_faithful, assert_regression...)
        │
        ├── Local path: sentence-transformers (no API call)
        │   cosine similarity → pass/fail
        │
        └── LLM Judge path: your chosen provider
            OpenAI / Anthropic / Groq / Ollama
            JSON response → score + reason → pass/fail
```

## Assertions

| Assertion | What it checks | Uses API? |
|-----------|---------------|-----------|
| `assert_faithful` | Every factual claim in output is supported by source | Yes |
| `assert_no_hallucination` | Output contains no invented facts not in source | Yes |
| `assert_tone` | Output matches an expected tone (freeform string) | Yes |
| `assert_semantic_similarity` | Cosine similarity between output and expected text | No |
| `assert_contains_claim` | Output semantically contains a given claim | Yes |
| `assert_safe` | Output contains no harmful or offensive content | Yes |
| `assert_language` | Output is written in the expected language | Yes |
| `assert_regression` | Output is not worse than a baseline (similarity + quality) | Yes |

## Configuration

### Environment variables

```bash
export LLM_JUDGE_PROVIDER=openai       # or anthropic, groq, ollama
export LLM_JUDGE_MODEL=gpt-4o-mini     # optional, defaults to provider best
export OPENAI_API_KEY=sk-...           # set for your chosen provider
```

### conftest.py

```python
from pytest_llm import pytest_configure_judge

pytest_configure_judge(provider="anthropic", model="claude-haiku-4-5-20251001")
```

### CLI options

```bash
pytest --llm-judge-provider=anthropic --llm-judge-model=claude-haiku-4-5-20251001
pytest --llm-report   # print Rich summary table after tests
```

## CI/CD with GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -v --tb=short
```

## Provider support

| Provider | Default Model | Env var for API key |
|----------|---------------|---------------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Groq | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Ollama | `llama3` | (local, no key needed) |

## Works with langgraph-replay

`pytest-llm` integrates with [langgraph-replay](https://github.com/Sushit-prog/langgraph-replay) for tracing and replaying LangGraph agent sessions during evaluation.

```bash
langgraph-replay blame session_abc --eval
```
