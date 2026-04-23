from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """Structured JSONL logger for benchmark runs."""

    def __init__(
        self,
        out_dir: str | Path,
        mode: str,
        dataset: str,
        log_level: str = "info",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.dataset = dataset
        self._level = log_level.lower()
        self.events_path = self.out_dir / "events.jsonl"

        self._logger = logging.getLogger(f"benchmark.{self.run_id}")
        self._logger.handlers.clear()
        self._logger.setLevel(logging.DEBUG if self._level == "debug" else logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def _should_log(self, level: str) -> bool:
        if self._level == "debug":
            return True
        return level != "debug"

    def emit(self, event: str, *, level: str = "info", **payload: Any) -> None:
        if not self._should_log(level):
            return
        row = {
            "ts": _utc_now(),
            "event": event,
            "level": level,
            "run_id": self.run_id,
            "mode": self.mode,
            "dataset": self.dataset,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self._level == "debug":
            self._logger.info(line)
        elif event in {"run_start", "run_end", "checkpoint_saved"}:
            self._logger.info(line)

    def run_start(self, *, total_examples: int, reflexion_attempts: int) -> None:
        self.emit(
            "run_start",
            total_examples=total_examples,
            reflexion_attempts=reflexion_attempts,
        )

    def sample_start(self, *, agent_type: str, index: int, total: int, qid: str) -> None:
        self.emit("sample_start", agent_type=agent_type, index=index, total=total, qid=qid, level="debug")

    def attempt_start(self, *, agent_type: str, qid: str, attempt_id: int) -> None:
        self.emit("attempt_start", agent_type=agent_type, qid=qid, attempt_id=attempt_id, level="debug")

    def llm_call_end(
        self,
        *,
        call_type: str,
        agent_type: str,
        qid: str,
        attempt_id: int,
        model: str,
        latency_ms: int,
        tokens: int,
        retry_count: int = 0,
    ) -> None:
        self.emit(
            "llm_call_end",
            level="debug",
            call_type=call_type,
            agent_type=agent_type,
            qid=qid,
            attempt_id=attempt_id,
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            retry_count=retry_count,
        )

    def sample_end(
        self,
        *,
        agent_type: str,
        index: int,
        total: int,
        qid: str,
        is_correct: bool,
        attempts: int,
        token_estimate: int,
        latency_ms: int,
        elapsed_ms: int,
    ) -> None:
        self.emit(
            "sample_end",
            agent_type=agent_type,
            index=index,
            total=total,
            qid=qid,
            is_correct=is_correct,
            attempts=attempts,
            token_estimate=token_estimate,
            latency_ms=latency_ms,
            elapsed_ms=elapsed_ms,
        )

    def sample_error(self, *, agent_type: str, index: int, total: int, qid: str, error: str) -> None:
        self.emit("sample_error", level="info", agent_type=agent_type, index=index, total=total, qid=qid, error=error)

    def checkpoint_saved(self, *, file_path: str, records: int) -> None:
        self.emit("checkpoint_saved", level="info", file_path=file_path, records=records)

    def run_end(
        self,
        *,
        succeeded: int,
        failed: int,
        total_records: int,
        elapsed_ms: int,
    ) -> None:
        self.emit(
            "run_end",
            succeeded=succeeded,
            failed=failed,
            total_records=total_records,
            elapsed_ms=elapsed_ms,
        )
