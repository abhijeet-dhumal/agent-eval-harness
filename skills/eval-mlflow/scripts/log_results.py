#!/usr/bin/env python3
"""Log eval run results to MLflow.

Reads summary.yaml and run_result.json, logs params, metrics,
artifacts, per-case results table, and creates the main orchestrator
trace from stdout.log.  Also links all experiment traces to the run.

Usage:
    python3 ${CLAUDE_SKILL_DIR}/scripts/log_results.py \\
        --run-id <id> \\
        --config eval.yaml
"""

import agent_eval._bootstrap  # noqa: F401 — auto-activate venv

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

try:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
except ImportError:
    print("MLflow not installed. Install with: pip install 'mlflow[genai]'",
          file=sys.stderr)
    sys.exit(0)

from agent_eval.config import EvalConfig
from agent_eval.mlflow.experiment import resolve_tracking_uri


# ── Trace builder (extracted to agent_eval/mlflow/trace_builder.py) ──
from agent_eval.mlflow.trace_builder import build_trace, log_trace
from agent_eval.mlflow.trajectory_trace import build_trace_from_trajectory


def _log_eval_dataset_lineage(config, per_case: dict) -> None:
    """Log eval case manifest as an MLflow dataset input."""
    if not per_case:
        return
    try:
        import pandas as pd
        import mlflow.data

        rows = [{"case_id": case_id} for case_id in sorted(per_case)]
        dataset_path = config.dataset.path if config.dataset else "eval-cases"
        ds = mlflow.data.from_pandas(
            pd.DataFrame(rows),
            source=str(dataset_path),
            name=f"{config.name}-eval-cases",
        )
        mlflow.log_input(ds, context="evaluation")
    except Exception as exc:
        print(f"WARNING: failed to log eval dataset lineage: {exc}", file=sys.stderr)


def _resolve_external_model_id(parent_run_id: str, run_result: dict, run_id: str) -> str | None:
    """Register a lightweight external model reference on the training run."""
    base_model = run_result.get("model")
    if not base_model:
        return None
    try:
        params = {"base_model": str(base_model)}
        if run_result.get("ray_job_id"):
            params["ray_job_id"] = str(run_result["ray_job_id"])
        model_info = mlflow.create_external_model(
            name=f"grpo-{run_id[:48]}",
            source_run_id=parent_run_id,
            params=params,
        )
        return model_info.model_id
    except Exception as exc:
        print(f"WARNING: failed to create external model reference: {exc}", file=sys.stderr)
        return None


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parent-run-id",
        default=None,
        help="NeMo RL training run id; eval results log to a nested child run",
    )
    parser.add_argument(
        "--lineage-ray-job-id",
        default=None,
        help="Ray job id tag for lineage (grpo-gate-YYYYMMDD-HHMMSS)",
    )
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    runs_base = Path(os.environ.get("AGENT_EVAL_RUNS_DIR", "eval/runs"))
    runs_dir = runs_base / config.eval_name()
    run_dir = runs_dir / args.run_id

    # Load summary
    summary_path = run_dir / "summary.yaml"
    if not summary_path.exists():
        print(f"ERROR: no summary found at {summary_path}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = yaml.safe_load(f) or {}

    # Load execution metadata
    run_result = {}
    run_result_path = run_dir / "run_result.json"
    if run_result_path.exists():
        with open(run_result_path) as f:
            run_result = json.load(f)

    # Set experiment
    experiment_name = config.mlflow.experiment or config.name
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Resolve experiment ID
    exp = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id if exp else "0"

    external_model_id = None
    if args.parent_run_id:
        external_model_id = _resolve_external_model_id(
            args.parent_run_id, run_result, args.run_id
        )

    run_kwargs: dict = {"run_name": args.run_id}
    if args.parent_run_id:
        run_kwargs["run_name"] = f"eval-{args.run_id}"
        run_kwargs["tags"] = {"mlflow.parentRunId": args.parent_run_id}

    with mlflow.start_run(**run_kwargs) as run:
        mlflow_run_id = run.info.run_id

        mlflow.set_tag("source", "agent-eval-harness")
        if args.parent_run_id:
            mlflow.set_tag("lineage_mode", "child_of_training_run")
            mlflow.set_tag("training_run_id", args.parent_run_id)
            mlflow.set_tag("harness_run_id", args.run_id)
            if external_model_id:
                mlflow.set_tag("mlflow.model.id", external_model_id)
        if args.lineage_ray_job_id:
            mlflow.set_tag("ray_job_id", args.lineage_ray_job_id)

        # ── Params ───────────────────────────────────────────────
        params = {
            "skill": config.resolve_skill(),
            "eval_name": config.eval_name(),
            "runner": config.runner.type,
            "run_id": args.run_id,
            "model": run_result.get("model", ""),
        }
        if run_result.get("subagent_model"):
            params["subagent_model"] = run_result["subagent_model"]
        if run_result.get("agent"):
            params["agent"] = run_result["agent"]
        for key, value in params.items():
            if value:
                mlflow.log_param(key, value)

        # ── Execution metrics ────────────────────────────────────
        if run_result.get("duration_s"):
            mlflow.log_metric("duration_s", run_result["duration_s"])
        if run_result.get("cost_usd"):
            mlflow.log_metric("cost_usd", run_result["cost_usd"])
        if run_result.get("num_turns"):
            mlflow.log_metric("num_turns", run_result["num_turns"])
        token_usage = run_result.get("token_usage", {})
        if token_usage:
            for key in ("input", "output", "cache_read", "cache_create"):
                val = token_usage.get(key)
                if val is not None:
                    mlflow.log_metric(f"tokens/{key}", val)

        # ── Per-model cost and token breakdown ───────────────────
        per_model = run_result.get("per_model_usage", {})
        if per_model:
            import re
            for model_name, stats in per_model.items():
                # Sanitize model name for MLflow metric keys:
                # only alphanumerics, underscores, dashes, periods, spaces,
                # colons, and slashes are allowed.
                safe_name = re.sub(r"[^A-Za-z0-9_\-\. :/]", "-", model_name)
                prefix = f"model/{safe_name}"
                if stats.get("cost_usd") is not None:
                    mlflow.log_metric(f"{prefix}/cost_usd", stats["cost_usd"])
                for key in ("input", "output", "cache_read", "cache_create"):
                    val = stats.get(key)
                    if val is not None:
                        mlflow.log_metric(f"{prefix}/tokens/{key}", val)

        # ── Judge metrics ────────────────────────────────────────
        judges = summary.get("judges", {})
        metric_count = 0
        for judge_name, agg in judges.items():
            if isinstance(agg, dict):
                if agg.get("pass_rate") is not None:
                    mlflow.log_metric(f"{judge_name}/pass_rate", agg["pass_rate"])
                    metric_count += 1
                if agg.get("mean") is not None:
                    mlflow.log_metric(f"{judge_name}/mean", agg["mean"])
                    metric_count += 1

        # ── Tags ─────────────────────────────────────────────────
        has_regressions = False
        if config.thresholds:
            for judge_name, threshold in config.thresholds.items():
                agg = judges.get(judge_name, {})
                if not isinstance(agg, dict):
                    continue
                if "min_pass_rate" in threshold:
                    rate = agg.get("pass_rate")
                    if rate is not None and rate < threshold["min_pass_rate"]:
                        has_regressions = True
                if "min_mean" in threshold:
                    mean = agg.get("mean")
                    if mean is not None and mean < threshold["min_mean"]:
                        has_regressions = True
        mlflow.set_tag("regressions_detected", "yes" if has_regressions else "no")
        mlflow.set_tag("num_judges", str(len(judges)))
        for tag_key, tag_value in (config.mlflow.tags or {}).items():
            mlflow.set_tag(tag_key, str(tag_value))

        # ── Artifacts ────────────────────────────────────────────
        if summary_path.exists():
            mlflow.log_artifact(str(summary_path))

        # Log input files for from-traces extraction.
        for name in ("batch.yaml", "case_order.yaml"):
            p = run_dir / name
            if p.exists():
                mlflow.log_artifact(str(p), "inputs")
        cases_dir = run_dir / "cases"
        if cases_dir.is_dir():
            for case_dir in sorted(cases_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                inp = case_dir / "input.yaml"
                if inp.exists():
                    mlflow.log_artifact(str(inp), f"inputs/{case_dir.name}")
                traj = case_dir / "trajectory.json"
                if traj.exists():
                    mlflow.log_artifact(str(traj), f"cases/{case_dir.name}")
                case_artifacts = case_dir / "artifacts"
                if case_artifacts.is_dir():
                    mlflow.log_artifacts(
                        str(case_artifacts), f"cases/{case_dir.name}/artifacts"
                    )
                for verifier_name in ("reward.json", "judges.json", "reward.txt"):
                    verifier_path = case_dir / "verifier" / verifier_name
                    if verifier_path.exists():
                        mlflow.log_artifact(
                            str(verifier_path),
                            f"cases/{case_dir.name}/verifier",
                        )

        # ── Per-case results table ───────────────────────────────
        per_case = summary.get("per_case", {})
        _log_eval_dataset_lineage(config, per_case)
        if per_case:
            table_rows = []
            for case_id, case_results in per_case.items():
                if not isinstance(case_results, dict):
                    continue
                for judge_name, result in case_results.items():
                    if not isinstance(result, dict):
                        continue
                    table_rows.append({
                        "case_id": case_id,
                        "judge": judge_name,
                        "value": result.get("value"),
                        "rationale": str(result.get("rationale", ""))[:500],
                    })
            if table_rows:
                columns = {}
                for key in table_rows[0]:
                    columns[key] = [row[key] for row in table_rows]
                mlflow.log_table(columns, artifact_file="per_case_results.json")

    # ── Find existing execution traces ────────────────────────────
    # Execution traces are created during skill execution by the trace
    # interceptor. They have eval_run_id tags matching the case IDs.
    # We prefer these over synthetic traces built from stdout.log
    # because they already exist in the DB (no async queue issues).
    main_trace_id = None
    case_trace_map = {}  # case_id -> trace_id
    trace_ids = []

    # TODO: paginate via page_token for experiments with >500 unlinked traces
    try:
        all_traces = client.search_traces(experiment_ids=[experiment_id],
                                          max_results=500)
        for t in all_traces:
            tags = t.info.tags or {}
            eval_id = tags.get("eval_run_id", "")
            existing_run = tags.get("mlflow.runId")
            # Match unlinked traces whose eval_run_id matches a case ID
            if eval_id and not existing_run:
                if eval_id not in case_trace_map:
                    case_trace_map[eval_id] = t.info.trace_id
                trace_ids.append(t.info.trace_id)
    except Exception as e:
        print(f"WARNING: failed to search traces: {e}", file=sys.stderr)

    # Build synthetic traces from stdout.log for cases not already covered
    # by existing execution traces.
    exec_mode = run_result.get("execution_mode", "batch")
    if exec_mode in ("case", "prompt", "harbor-grpo"):
        cases_dir = run_dir / "cases"
        if cases_dir.exists():
            for case_dir in sorted(d for d in cases_dir.iterdir() if d.is_dir()):
                case_id = case_dir.name
                if case_id in case_trace_map:
                    continue
                case_result = run_result.get("per_case", {}).get(case_id, run_result)
                _skill = config.resolve_skill()
                trace_name = f"{_skill} ({case_id})" if _skill else case_id
                trace_dict = None
                case_stdout = case_dir / "stdout.log"
                if case_stdout.exists():
                    trace_dict = build_trace(
                        case_stdout, case_result, case_id, experiment_id, trace_name=trace_name
                    )
                else:
                    for trajectory_path in (
                        case_dir / "trajectory.json",
                        case_dir / "agent" / "trajectory.json",
                    ):
                        if trajectory_path.exists():
                            trace_dict = build_trace_from_trajectory(
                                trajectory_path,
                                case_result,
                                case_id,
                                experiment_id,
                                trace_name=trace_name,
                            )
                            break
                if trace_dict:
                    tid = log_trace(trace_dict)
                    if tid:
                        case_trace_map[case_id] = tid
                        trace_ids.append(tid)
                        num_spans = len(trace_dict["data"]["spans"])
                        print(f"TRACE: {tid} ({num_spans} spans) — {case_id}")
    else:
        stdout_path = run_dir / "stdout.log"
        if stdout_path.exists() and run_result:
            _skill = config.resolve_skill()
            trace_name = f"{_skill} ({args.run_id})" if _skill else config.eval_name()
            trace_dict = build_trace(stdout_path, run_result, args.run_id,
                                     experiment_id, trace_name=trace_name)
            if trace_dict:
                main_trace_id = log_trace(trace_dict)
                if main_trace_id:
                    trace_ids.append(main_trace_id)
                    num_spans = len(trace_dict["data"]["spans"])
                    duration_s = run_result.get("duration_s", 0)
                    print(f"TRACE: {main_trace_id} ({num_spans} spans, {duration_s:.0f}s)")

    # Flush async queue so traces are committed before linking/feedback.
    if trace_ids:
        try:
            from mlflow.tracing.export.async_export_queue import AsyncExportQueue
            AsyncExportQueue.get_instance().flush(timeout_sec=30)
        except Exception as e:
            print(f"WARNING: trace export flush failed: {e}", file=sys.stderr)

    if not main_trace_id and case_trace_map:
        main_trace_id = next(iter(case_trace_map.values()))

    # ── Link traces to run (must happen before feedback) ─────────
    try:
        if trace_ids:
            client.link_traces_to_run(run_id=mlflow_run_id, trace_ids=trace_ids)
            for tid in trace_ids:
                client.set_trace_tag(tid, "mlflow.runId", mlflow_run_id)
            print(f"LINKED: {len(trace_ids)} traces to run {mlflow_run_id}")
    except Exception as e:
        print(f"WARNING: failed to link traces: {e}", file=sys.stderr)

    # ── Attach judge feedback to traces (populates Quality tab) ──
    feedback_count = 0

    _TRUE_STRS = {"pass", "true", "yes", "y", "1", "ok", "success"}
    _FALSE_STRS = {"fail", "false", "no", "n", "0", "error", "failure"}

    def _to_feedback_value(v):
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in _TRUE_STRS:
                return 1.0
            if s in _FALSE_STRS:
                return 0.0
            try:
                return float(s)
            except ValueError:
                return None
        return None

    for case_id, case_results in per_case.items():
        trace_id = case_trace_map.get(case_id, main_trace_id)
        if not trace_id or not isinstance(case_results, dict):
            continue
        for judge_name, result in case_results.items():
            if not isinstance(result, dict):
                continue
            value = result.get("value")
            if value is None:
                continue
            fb_value = _to_feedback_value(value)
            if fb_value is None:
                continue
            try:
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name=judge_name,
                    value=fb_value,
                    rationale=str(result.get("rationale", ""))[:500],
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.CODE,
                        source_id=f"eval/{judge_name}",
                    ),
                )
                feedback_count += 1
            except Exception as e:
                print(f"WARNING: failed to log feedback for {case_id}/{judge_name}: {e}",
                      file=sys.stderr)

    print(f"EXPERIMENT: {experiment_name}")
    print(f"RUN: {mlflow_run_id}")
    if args.parent_run_id:
        print(f"PARENT: {args.parent_run_id}")
    if external_model_id:
        print(f"MODEL: {external_model_id}")
    print(f"PARAMS: {len(params)}")
    print(f"METRICS: {metric_count}")
    print(f"TABLE: per_case_results ({len(per_case)} cases)")
    if feedback_count:
        print(f"FEEDBACK: {feedback_count} assessments attached to traces")


if __name__ == "__main__":
    main()
