"""Generate NeMo Gym training JSONL from Harbor task packages.

Converts Harbor task packages (from :func:`agent_eval.harbor.tasks.generate_tasks`)
into a JSONL file formatted for NeMo Gym's harbor_agent training data loader.

Each line is a JSON object with:
  - ``instance_id``: ``"<dataset_alias>::<task_dir_name>"``
  - ``responses_create_params``: ``{"input": []}``
  - ``agent_ref``: ``{"name": "<agent_ref>"}``

The ``dataset_alias`` must match the key in ``harbor_datasets`` in the
NeMo Gym agent config (e.g., ``harbor_rfe_openshift.yaml``). The task_dir_name
is the directory name under the tasks dir (e.g., ``case-001-my-task``).

Usage (CLI):
    python3 -m agent_eval.harbor.training \\
        --tasks-dir tasks/rfe-cases \\
        --out data/train.jsonl \\
        --repeat 4 --shuffle

Usage (API):
    from agent_eval.harbor.training import generate_training_jsonl
    generate_training_jsonl(
        tasks_dir=Path("tasks/rfe-cases"),
        out_path=Path("data/train.jsonl"),
        dataset_alias="rfe",
        repeat=4,
        shuffle=True,
    )
"""

import argparse
import json
import random
import sys
from pathlib import Path


def _build_entry(task_dir: Path, dataset_alias: str, agent_ref: str) -> dict:
    return {
        "instance_id": f"{dataset_alias}::{task_dir.name}",
        "responses_create_params": {"input": []},
        "agent_ref": {"name": agent_ref},
    }


def generate_training_jsonl(
    tasks_dir: Path,
    out_path: Path,
    *,
    dataset_alias: str = "rfe",
    agent_ref: str = "harbor_agent",
    repeat: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[dict]:
    """Generate a NeMo Gym training JSONL from Harbor task packages.

    Args:
        tasks_dir: Directory containing Harbor task packages (each with task.toml).
        out_path: Output JSONL file path.
        dataset_alias: Key in harbor_datasets config that maps to this dataset.
        agent_ref: NeMo Gym agent reference name.
        repeat: Repeat each case N times (useful for GRPO variance).
        shuffle: Randomize entry order after repeating.
        seed: Random seed for shuffle reproducibility.

    Returns:
        List of generated entries (dicts).
    """
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Task packages not found: {tasks_dir}")

    task_dirs = sorted(
        d for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "task.toml").is_file()
    )
    if not task_dirs:
        raise ValueError(f"No task packages with task.toml found in {tasks_dir}")

    entries = [_build_entry(d, dataset_alias, agent_ref) for d in task_dirs] * repeat
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NeMo Gym training JSONL from Harbor task packages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks-dir", required=True,
                        help="Path to Harbor task packages directory")
    parser.add_argument("--out", required=True,
                        help="Output JSONL path")
    parser.add_argument("--dataset-alias", default="rfe",
                        help="harbor_datasets key in agent config (default: rfe)")
    parser.add_argument("--agent-ref", default="harbor_agent",
                        help="NeMo Gym agent reference name (default: harbor_agent)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each case N times (default: 1)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the output order")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffle (default: None)")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    out_path = Path(args.out)

    try:
        entries = generate_training_jsonl(
            tasks_dir=tasks_dir,
            out_path=out_path,
            dataset_alias=args.dataset_alias,
            agent_ref=args.agent_ref,
            repeat=args.repeat,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    task_count = len(entries) // args.repeat
    print(f"Wrote {len(entries)} entries → {out_path}")
    print(f"  {task_count} unique cases × {args.repeat} repeat(s)")
    for e in entries[:3]:
        print(f"  {e['instance_id']}")
    if len(entries) > 3:
        print(f"  ... ({len(entries) - 3} more)")


if __name__ == "__main__":
    main()
