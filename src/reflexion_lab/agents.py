from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
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

    def run(self, example: QAExample) -> RunRecord:
        actor_answer, evaluator, reflector = self._runtime()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, judge_tokens, judge_latency = evaluator(example, answer)
            token_estimate = actor_tokens + judge_tokens
            latency_ms = actor_latency + judge_latency
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, reflection_tokens, reflection_latency = reflector(example, attempt_id, judge, answer)
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                trace.reflection = reflection
                trace.token_estimate += reflection_tokens
                trace.latency_ms += reflection_latency
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
