"""Build MLflow traces from Harbor ATIF trajectory.json files."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_eval.mlflow.trace_builder import iso_to_ns, make_span, summarize_tool_input


def build_trace_from_trajectory(
    trajectory_path: Path,
    run_result: dict,
    run_id: str,
    experiment_id: str,
    trace_name: str = "",
) -> dict | None:
    """Convert a Harbor ``agent/trajectory.json`` file into an MLflow trace dict."""
    if not trajectory_path.is_file():
        return None

    try:
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    steps = payload.get("steps") or []
    if not steps:
        return None

    trace_id = uuid.uuid4().hex
    root_span_id = uuid.uuid4().bytes[:8].hex()

    prompt = ""
    final_response = ""
    for step in steps:
        if step.get("source") == "user" and not prompt:
            prompt = str(step.get("message") or "").strip()
        if step.get("source") == "agent":
            message = str(step.get("message") or "").strip()
            if message:
                final_response = message

    first_ts = steps[0].get("timestamp")
    last_ts = steps[-1].get("timestamp")
    if first_ts and last_ts:
        trace_start = iso_to_ns(first_ts)
        trace_end = iso_to_ns(last_ts)
    else:
        now = datetime.now(tz=timezone.utc)
        trace_start = int(now.timestamp() * 1e9)
        trace_end = trace_start + int(1e9)

    spans = [
        make_span(
            trace_id,
            None,
            trace_name or f"harbor ({run_id})",
            "AGENT",
            trace_start,
            trace_end,
            inputs={"prompt": prompt[:500]} if prompt else None,
            outputs={"response": final_response[:500]} if final_response else None,
        )
    ]
    root_span_id = spans[0]["span_id"]

    step_start = trace_start
    for index, step in enumerate(steps, start=1):
        step_end = trace_end
        if step.get("timestamp"):
            try:
                step_end = iso_to_ns(step["timestamp"])
            except Exception:
                step_end = step_start + int(1e8)

        source = step.get("source", "unknown")
        if source == "user":
            spans.append(
                make_span(
                    trace_id,
                    root_span_id,
                    f"user-{index}",
                    "LLM",
                    step_start,
                    step_end,
                    inputs={"message": str(step.get("message") or "")[:500]},
                )
            )
            step_start = step_end
            continue

        message = str(step.get("message") or "").strip()
        if message:
            spans.append(
                make_span(
                    trace_id,
                    root_span_id,
                    f"agent-{index}",
                    "LLM",
                    step_start,
                    step_end,
                    outputs={"response": message[:500]},
                )
            )
            step_start = step_end

        for tool_call in step.get("tool_calls") or []:
            tool_name = tool_call.get("function_name") or tool_call.get("tool_use_name") or "tool"
            tool_input = tool_call.get("arguments") or {}
            if not isinstance(tool_input, dict):
                tool_input = {"input": str(tool_input)[:200]}
            tool_end = step_end
            spans.append(
                make_span(
                    trace_id,
                    root_span_id,
                    tool_name,
                    "TOOL",
                    step_start,
                    tool_end,
                    inputs=summarize_tool_input(tool_name, tool_input),
                )
            )
            step_start = tool_end

    duration_ms = max(int((trace_end - trace_start) / 1e6), 1)
    trace_metadata = {
        "mlflow.traceName": trace_name or f"harbor ({run_id})",
    }
    token_usage = run_result.get("token_usage") or {}
    if token_usage:
        trace_metadata["mlflow.trace.tokenUsage"] = json.dumps({
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0),
            "total_tokens": token_usage.get("input", 0) + token_usage.get("output", 0),
        })

    return {
        "info": {
            "trace_id": trace_id,
            "trace_location": {
                "type": "MLFLOW_EXPERIMENT",
                "mlflow_experiment": {"experiment_id": experiment_id},
            },
            "request_time": datetime.fromtimestamp(
                trace_start / 1e9, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "trace_metadata": trace_metadata,
            "state": "OK",
            "execution_duration_ms": duration_ms,
            "request_preview": prompt[:200],
            "response_preview": final_response[:200],
            "tags": {
                "eval_run_id": run_id,
                "source": "harbor-trajectory",
                "mlflow.traceName": trace_name or f"harbor ({run_id})",
            },
        },
        "data": {"spans": spans},
    }
