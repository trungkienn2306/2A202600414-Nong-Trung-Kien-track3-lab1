from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime_mode: Literal["mock", "real"] = "mock"

    def _runtime(self):
        if self.runtime_mode == "real":
            from .real_runtime import actor_answer, evaluator, reflector
        else:
            from .mock_runtime import actor_answer, evaluator, reflector
        return actor_answer, evaluator, reflector

    def run(self, example: QAExample, *, logger: Any | None = None, runtime_options: dict[str, Any] | None = None) -> RunRecord:
        actor_answer, evaluator, reflector = self._runtime()
        runtime_options = runtime_options or {}
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            if logger is not None:
                logger.attempt_start(agent_type=self.agent_type, qid=example.qid, attempt_id=attempt_id)
            answer, actor_tokens, actor_latency = actor_answer(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
                logger=logger,
                max_retries=runtime_options.get("max_retries", 2),
                retry_backoff_seconds=runtime_options.get("retry_backoff_seconds", 1.0),
            )
            judge, judge_tokens, judge_latency = evaluator(
                example,
                answer,
                logger=logger,
                agent_type=self.agent_type,
                attempt_id=attempt_id,
                max_retries=runtime_options.get("max_retries", 2),
                retry_backoff_seconds=runtime_options.get("retry_backoff_seconds", 1.0),
            )
            token_estimate = actor_tokens + judge_tokens
            latency_ms = actor_latency + judge_latency
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if logger is not None:
                logger.emit(
                    "attempt_end",
                    level="debug",
                    agent_type=self.agent_type,
                    qid=example.qid,
                    attempt_id=attempt_id,
                    score=judge.score,
                    token_estimate=trace.token_estimate,
                    latency_ms=trace.latency_ms,
                    reflection_next_strategy="",
                )
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, reflection_tokens, reflection_latency = reflector(
                    example,
                    attempt_id,
                    judge,
                    answer,
                    logger=logger,
                    agent_type=self.agent_type,
                    max_retries=runtime_options.get("max_retries", 2),
                    retry_backoff_seconds=runtime_options.get("retry_backoff_seconds", 1.0),
                )
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.reflection = reflection
                trace.token_estimate += reflection_tokens
                trace.latency_ms += reflection_latency
            if logger is not None and trace.reflection is not None:
                logger.emit(
                    "attempt_reflection",
                    level="debug",
                    agent_type=self.agent_type,
                    qid=example.qid,
                    attempt_id=attempt_id,
                    reflection_next_strategy=trace.reflection.next_strategy,
                )
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, runtime_mode: Literal["mock", "real"] = "mock") -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime_mode=runtime_mode)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime_mode: Literal["mock", "real"] = "mock") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime_mode=runtime_mode)
