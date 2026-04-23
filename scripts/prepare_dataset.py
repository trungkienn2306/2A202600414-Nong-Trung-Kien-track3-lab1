from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from datasets import load_dataset

app = typer.Typer(add_completion=False)


def _difficulty(level: str) -> str:
    if level == "easy":
        return "easy"
    if level == "medium":
        return "medium"
    return "hard"


@app.command()
def main(
    out: str = "data/hotpot_100.json",
    limit: int = 100,
    seed: int = 42,
    max_context_chunks: int = 2,
    max_sentences_per_chunk: int = 3,
) -> None:
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    records: list[dict] = []
    for idx in indices:
        row = ds[idx]
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        context = []
        for i in range(min(max_context_chunks, len(titles))):
            sentences = sentences_list[i][:max_sentences_per_chunk]
            context.append({"title": titles[i], "text": " ".join(sentences).strip()})
        if not context:
            continue
        records.append(
            {
                "qid": str(row["id"]),
                "difficulty": _difficulty(row.get("level", "hard")),
                "question": row["question"],
                "gold_answer": row["answer"],
                "context": context,
            }
        )
        if len(records) >= limit:
            break

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Saved {len(records)} examples to {out_path}")


if __name__ == "__main__":
    app()
