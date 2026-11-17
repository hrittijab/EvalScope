
"""Generate a reproducible synthetic EvalScope demo dataset."""

import json
import random
from pathlib import Path


def generate_demo(
    output_path="data/demo_experiment.json",
    seed=42,
):
    rng = random.Random(seed)

    categories = [
        "workflow_hijacking",
        "authority_impersonation",
        "encoded_payload",
        "instruction_override",
    ]

    modes = [
        "baseline",
        "prompt_guard",
        "authorization",
        "layered",
    ]

    success_probabilities = {
        "baseline": 0.65,
        "prompt_guard": 0.35,
        "authorization": 0.12,
        "layered": 0.05,
    }

    records = []

    for i in range(120):
        attack_id = f"DEMO-{i:04d}"
        source_id = f"SEED-{i // 4:03d}"
        category = categories[i % len(categories)]

        # Shared difficulty creates dependence between
        # configurations evaluated on the same attack.
        difficulty = rng.random()

        for mode in modes:
            probability = success_probabilities[mode]

            # A small fraction of runs are incomplete.
            incomplete = rng.random() < 0.08

            status = (
                rng.choice(["truncated", "timeout"])
                if incomplete
                else "completed"
            )

            records.append({
                "experiment_id": "synthetic-demo-v1",
                "attack_id": attack_id,
                "source_attack_id": source_id,
                "category": category,
                "mode": mode,
                "attack_success": (
                    difficulty < probability
                    if not incomplete
                    else False
                ),
                "status": status,
                "valid_trial": not incomplete,
                "duration_seconds": round(
                    rng.uniform(0.5, 8.0), 3
                ),
                "model": "synthetic-demo",
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    path = generate_demo()
    print(f"Generated synthetic dataset: {path}")
