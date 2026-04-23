from __future__ import annotations

import json
import os
import time
from functools import lru_cache
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


@lru_cache(maxsize=1)
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


def _with_retries(
    fn,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    logger: Any | None = None,
    log_payload: dict[str, Any] | None = None,
):
    last_error: Exception | None = None
    attempts = max_retries + 1
    for retry_count in range(attempts):
        try:
            started = time.perf_counter()
            result = fn()
            latency_ms = int((time.perf_counter() - started) * 1000)
            return result, retry_count, latency_ms
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if logger is not None:
                logger.emit(
                    "llm_call_retry",
                    level="info",
                    retry_count=retry_count,
                    max_retries=max_retries,
                    error=str(exc),
                    **(log_payload or {}),
                )
            if retry_count >= max_retries:
                break
            sleep_s = retry_backoff_seconds * (2**retry_count)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after retries: {last_error}") from last_error


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    *,
    logger: Any | None = None,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
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
    response, retry_count, latency_ms = _with_retries(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            timeout=60,
        ),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        logger=logger,
        log_payload={"call_type": "actor", "agent_type": agent_type, "qid": example.qid, "attempt_id": attempt_id, "model": model},
    )
    answer = _extract_text(response)
    tokens = _usage_tokens(response)
    if logger is not None:
        logger.llm_call_end(
            call_type="actor",
            agent_type=agent_type,
            qid=example.qid,
            attempt_id=attempt_id,
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            retry_count=retry_count,
        )
    return answer, tokens, latency_ms


def evaluator(
    example: QAExample,
    answer: str,
    *,
    logger: Any | None = None,
    agent_type: str = "unknown",
    attempt_id: int = 0,
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
) -> tuple[JudgeResult, int, int]:
    judge_model = _first_env("judge_model", "JUDGE_MODEL", required_label="judge_model/JUDGE_MODEL")
    user_prompt = (
        f"question: {example.question}\n"
        f"gold_answer: {example.gold_answer}\n"
        f"predicted_answer: {answer}\n"
        "Judge correctness and return strict JSON."
    )
    client = _client()
    response, retry_count, latency_ms = _with_retries(
        lambda: client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=60,
        ),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        logger=logger,
        log_payload={"call_type": "evaluator", "agent_type": agent_type, "qid": example.qid, "attempt_id": attempt_id, "model": judge_model},
    )
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
    tokens = _usage_tokens(response)
    if logger is not None:
        logger.llm_call_end(
            call_type="evaluator",
            agent_type=agent_type,
            qid=example.qid,
            attempt_id=attempt_id,
            model=judge_model,
            latency_ms=latency_ms,
            tokens=tokens,
            retry_count=retry_count,
        )
    return result, tokens, latency_ms


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    answer: str,
    *,
    logger: Any | None = None,
    agent_type: str = "reflexion",
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
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
    response, retry_count, latency_ms = _with_retries(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=60,
        ),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        logger=logger,
        log_payload={"call_type": "reflector", "agent_type": agent_type, "qid": example.qid, "attempt_id": attempt_id, "model": model},
    )
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
    tokens = _usage_tokens(response)
    if logger is not None:
        logger.llm_call_end(
            call_type="reflector",
            agent_type=agent_type,
            qid=example.qid,
            attempt_id=attempt_id,
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            retry_count=retry_count,
        )
    return reflection, tokens, latency_ms
