from __future__ import annotations
import json
import time
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.logging_utils import RunLogger
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)


def _fallback_record(agent_type: str, example, elapsed_ms: int, error_message: str) -> RunRecord:
    return RunRecord(
        qid=example.qid,
        question=example.question,
        gold_answer=example.gold_answer,
        agent_type=agent_type,
        predicted_answer=f"__ERROR__: {error_message[:200]}",
        is_correct=False,
        attempts=1,
        token_estimate=0,
        latency_ms=max(1, elapsed_ms),
        failure_mode="wrong_final_answer",
        reflections=[],
        traces=[],
    )


def _run_agent_with_progress(
    *,
    agent,
    agent_type: str,
    examples,
    logger: RunLogger,
    continue_on_error: bool,
    checkpoint_every: int,
    out_path: Path,
    runtime_options: dict,
) -> tuple[list[RunRecord], list[dict]]:
    records: list[RunRecord] = []
    failures: list[dict] = []
    total = len(examples)
    checkpoint_name = f"{agent_type}_runs.partial.jsonl"
    for idx, example in enumerate(examples, start=1):
        logger.sample_start(agent_type=agent_type, index=idx, total=total, qid=example.qid)
        started = time.perf_counter()
        try:
            record = agent.run(example, logger=logger, runtime_options=runtime_options)
            records.append(record)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.sample_end(
                agent_type=agent_type,
                index=idx,
                total=total,
                qid=example.qid,
                is_correct=record.is_correct,
                attempts=record.attempts,
                token_estimate=record.token_estimate,
                latency_ms=record.latency_ms,
                elapsed_ms=elapsed_ms,
            )
            print(
                f"[cyan]{agent_type}[/cyan] {idx}/{total} qid={example.qid} "
                f"correct={record.is_correct} attempts={record.attempts} "
                f"tokens={record.token_estimate} latency_ms={record.latency_ms}"
            )
        except Exception as exc:  # noqa: BLE001
            failure = {"agent_type": agent_type, "qid": example.qid, "index": idx, "error": str(exc)}
            failures.append(failure)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            fallback = _fallback_record(agent_type, example, elapsed_ms, str(exc))
            records.append(fallback)
            logger.sample_error(agent_type=agent_type, index=idx, total=total, qid=example.qid, error=str(exc))
            logger.sample_end(
                agent_type=agent_type,
                index=idx,
                total=total,
                qid=example.qid,
                is_correct=False,
                attempts=fallback.attempts,
                token_estimate=fallback.token_estimate,
                latency_ms=fallback.latency_ms,
                elapsed_ms=elapsed_ms,
            )
            print(f"[red]{agent_type}[/red] {idx}/{total} qid={example.qid} failed: {exc}")
            if not continue_on_error:
                raise
        if idx % checkpoint_every == 0:
            checkpoint_path = out_path / checkpoint_name
            save_jsonl(checkpoint_path, records)
            logger.checkpoint_saved(file_path=str(checkpoint_path), records=len(records))
    checkpoint_path = out_path / checkpoint_name
    save_jsonl(checkpoint_path, records)
    logger.checkpoint_saved(file_path=str(checkpoint_path), records=len(records))
    return records, failures


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    log_level: str = "info",
    checkpoint_every: int = 10,
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue processing remaining examples when one example fails.",
    ),
    max_retries: int = 2,
    retry_backoff_seconds: float = 1.0,
) -> None:
    if mode not in {"mock", "real"}:
        raise typer.BadParameter("mode must be either 'mock' or 'real'")
    if log_level not in {"info", "debug"}:
        raise typer.BadParameter("log_level must be either 'info' or 'debug'")
    if checkpoint_every < 1:
        raise typer.BadParameter("checkpoint_every must be >= 1")
    if max_retries < 0:
        raise typer.BadParameter("max_retries must be >= 0")
    examples = load_dataset(dataset)
    out_path = Path(out_dir)
    logger = RunLogger(out_path, mode=mode, dataset=Path(dataset).name, log_level=log_level)
    runtime_options = {"max_retries": max_retries, "retry_backoff_seconds": retry_backoff_seconds}
    started = time.perf_counter()
    logger.run_start(total_examples=len(examples), reflexion_attempts=reflexion_attempts)
    react = ReActAgent(runtime_mode=mode)
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts, runtime_mode=mode)
    react_records, react_failures = _run_agent_with_progress(
        agent=react,
        agent_type="react",
        examples=examples,
        logger=logger,
        continue_on_error=continue_on_error,
        checkpoint_every=checkpoint_every,
        out_path=out_path,
        runtime_options=runtime_options,
    )
    reflexion_records, reflexion_failures = _run_agent_with_progress(
        agent=reflexion,
        agent_type="reflexion",
        examples=examples,
        logger=logger,
        continue_on_error=continue_on_error,
        checkpoint_every=checkpoint_every,
        out_path=out_path,
        runtime_options=runtime_options,
    )
    all_records = react_records + reflexion_records
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    failures = react_failures + reflexion_failures
    failures_path = out_path / "failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.run_end(
        succeeded=len(all_records),
        failed=len(failures),
        total_records=(len(examples) * 2),
        elapsed_ms=elapsed_ms,
    )
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(f"[yellow]Saved[/yellow] {failures_path}")
    avg_sample_ms = (elapsed_ms / (len(examples) * 2)) if examples else 0
    print(
        f"[bold]Run summary[/bold] processed={len(examples) * 2} "
        f"succeeded={len(all_records)} failed={len(failures)} "
        f"elapsed_ms={elapsed_ms} avg_sample_ms={round(avg_sample_ms, 2)}"
    )
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
