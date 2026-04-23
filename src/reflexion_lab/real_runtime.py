from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()


def _get_env(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name) or fallback
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _first_env(*names: str, required_label: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable: {required_label}")


def _client() -> OpenAI:
    base_url = _get_env("DEFAULT_BASE_URL")
    api_key = _first_env("default_api_key", "DEFAULT_API_KEY", required_label="default_api_key/DEFAULT_API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key)


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    choices = getattr(response, "choices", [])
    if choices:
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", "") if msg else ""
        if isinstance(content, str):
            return content.strip()
    return ""


def _usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is not None:
        return int(total_tokens)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt_tokens + completion_tokens)


def _context_block(example: QAExample) -> str:
    blocks = []
    for idx, chunk in enumerate(example.context, start=1):
        blocks.append(f"[{idx}] {chunk.title}\n{chunk.text}")
    return "\n\n".join(blocks)


def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, int, int]:
    model = _first_env("default_model", "DEFAULT_MODEL", required_label="default_model/DEFAULT_MODEL")
    memory_text = "\n".join(f"- {item}" for item in reflection_memory[-5:]) if reflection_memory else "(none)"
    user_prompt = (
        f"attempt_id: {attempt_id}\n"
        f"agent_type: {agent_type}\n"
        f"question: {example.question}\n\n"
        f"context:\n{_context_block(example)}\n\n"
        f"reflection_memory:\n{memory_text}\n\n"
        "Return only the final answer."
    )

    client = _client()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    answer = _extract_text(response)
    return answer, _usage_tokens(response), latency_ms


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    judge_model = _first_env("judge_model", "JUDGE_MODEL", required_label="judge_model/JUDGE_MODEL")
    user_prompt = (
        f"question: {example.question}\n"
        f"gold_answer: {example.gold_answer}\n"
        f"predicted_answer: {answer}\n"
        "Judge correctness and return strict JSON."
    )
    client = _client()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    raw = _extract_text(response)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "score": 0,
            "reason": "Evaluator returned non-JSON output.",
            "missing_evidence": ["Could not parse evaluator output."],
            "spurious_claims": [answer] if answer else [],
        }
    result = JudgeResult.model_validate(data)
    return result, _usage_tokens(response), latency_ms


def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult, answer: str
) -> tuple[ReflectionEntry, int, int]:
    model = _first_env("default_model", "DEFAULT_MODEL", required_label="default_model/DEFAULT_MODEL")
    user_prompt = (
        f"attempt_id: {attempt_id}\n"
        f"question: {example.question}\n"
        f"context:\n{_context_block(example)}\n\n"
        f"failed_answer: {answer}\n"
        f"judge_result: {judge.model_dump_json()}\n"
        "Return strict JSON reflection."
    )
    client = _client()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    raw = _extract_text(response)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "attempt_id": attempt_id,
            "failure_reason": judge.reason,
            "lesson": "Need stricter grounding to context.",
            "next_strategy": "Resolve each hop explicitly and verify final entity against context.",
        }
    reflection = ReflectionEntry.model_validate(data)
    return reflection, _usage_tokens(response), latency_ms
