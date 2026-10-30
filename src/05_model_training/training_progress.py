"""Epoch progress and bounded, read-only views of active training journals."""

import json
import math
from pathlib import Path


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def epoch_progress(epoch_update, total_updates, windows_done, total_windows, elapsed):
    rate = epoch_update / elapsed if elapsed > 0 else None
    eta = max(0, total_updates - epoch_update) / rate if rate else None
    return {
        "epoch_update": epoch_update,
        "epoch_total_updates": total_updates,
        "epoch_windows_completed": windows_done,
        "epoch_total_windows": total_windows,
        "epoch_progress_percent": 100 * epoch_update / total_updates,
        "epoch_elapsed_seconds": elapsed,
        "epoch_updates_per_second": rate,
        "epoch_eta_seconds": eta,
        "eta_scope": "training_pass_only_excludes_validation",
    }


def journal_progress(journal, plan, tail_bytes=2 * 1024 * 1024):
    """Read a bounded tail; works for journals created before progress fields existed."""
    journal = Path(journal)
    with journal.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - tail_bytes))
        lines = stream.read().splitlines()
    events = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
        except (ValueError, UnicodeDecodeError):
            continue  # Partial lines at either end of an actively written file.
    terminal = events[-1].get("event") if events else "initializing"
    if terminal == "validation_progress":
        last = events[-1]
        completed = last["validation_tables_completed"]
        total = last["validation_total_tables"]
        return {
            "journal": str(journal),
            "latest_event": terminal,
            "validation_tables_completed": completed,
            "validation_total_tables": total,
            "message": f"Full validation | tables {completed:,}/{total:,} "
            f"({100 * completed / total:.1f}%) | validation ETA {duration(last['validation_eta_seconds'])}",
        }
    updates = [
        e for e in events if e.get("event") == "metrics" and e.get("epoch") is not None
    ]
    result = {"journal": str(journal), "latest_event": terminal}
    if not updates:
        message = "Waiting for the first optimizer update"
        if terminal == "run_error":
            message = f"Run stopped: {events[-1].get('error_type', 'error')}"
        elif terminal == "run_complete":
            message = "Run complete"
        return {**result, "message": message}
    last = updates[-1]
    total = last.get("epoch_total_updates", plan["updates_per_epoch"])
    epoch = last["epoch"]
    step = last["global_step"]
    local = last.get("epoch_update", step - (epoch - 1) * total)
    rate = last.get("epoch_updates_per_second")
    if rate is None:
        same_epoch = [e for e in updates if e["epoch"] == epoch]
        if len(same_epoch) > 1:
            first = same_epoch[0]
            elapsed = last["elapsed_seconds"] - first["elapsed_seconds"]
            rate = (step - first["global_step"]) / elapsed if elapsed > 0 else None
    eta = max(0, total - local) / rate if rate else None
    result = {
        **result,
        "epoch": epoch,
        "global_step": step,
        "epoch_update": local,
        "epoch_total_updates": total,
        "epoch_progress_percent": 100 * local / total,
        "epoch_updates_per_second": rate,
        "epoch_eta_seconds": eta,
        "eta_scope": "training_pass_only_excludes_validation",
        "message": f"Epoch {epoch} | update {local:,}/{total:,} ({100 * local / total:.1f}%) "
        f"| global {step:,} | training ETA {duration(eta)}"
        + (
            " | training pass complete; validation/checkpoint work may follow"
            if local >= total
            else ""
        ),
    }
    if terminal in ("run_complete", "run_error"):
        status = (
            "Run complete"
            if terminal == "run_complete"
            else f"Run stopped: {events[-1].get('error_type', 'error')}"
        )
        result["epoch_eta_seconds"] = None
        result["message"] = (
            f"{status} | last recorded epoch {epoch}, update {local:,}/{total:,} ({100 * local / total:.1f}%) | global {step:,}"
        )
    return result
