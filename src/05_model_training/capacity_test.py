"""Capacity diagnostic: full training and validation sweeps after every update."""

import argparse
import csv
import math
import sys
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from batching import make_batches, pack_records
from learning_rate import make_scheduler, scheduler_snapshot
from train import RWKVForecaster, architecture_report, launch_wsl, on_device
from training_log import emit_training_event, logged_training
from transform import (
    file_digest,
    load_config,
    location,
    read_artifact,
    run_context,
    save_artifact,
    signature,
    write_json,
)


class MasterAdamW:
    """FP32 optimizer state and master weights; bf16 forward/backward parameters."""

    def __init__(self, model, settings):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.masters = [
            torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.masters,
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            foreach=False,
        )

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.parameters:
            p.grad = None

    def step(self, gradient_clip):
        for master, p in zip(self.masters, self.parameters):
            master.grad = None if p.grad is None else p.grad.detach().float()
        norm = torch.nn.utils.clip_grad_norm_(
            self.masters, gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            for p, master in zip(self.parameters, self.masters):
                p.copy_(master)
        return float(norm)


@torch.no_grad()
def evaluate(model, batches, step, split, table_writer):
    started = perf_counter()
    was_training = model.training
    model.eval()
    points = 0
    pinball_sum = squared_sum = normalized_squared_sum = 0.0
    macro_loss = []
    try:
        for records in batches:
            prepared = on_device(pack_records(records))
            output = model.forward_encoded(model.encode_tokens(prepared), prepared)
            labels = prepared["labels"]
            valid = torch.isfinite(labels)
            target = labels.nan_to_num()
            scale = prepared["scale"][:, : prepared["targets"]]
            error = (target[..., None] - output.quantiles) / scale[..., None]
            levels = error.new_tensor(model.config.quantiles)
            pinball = torch.maximum(levels * error, (levels - 1) * error).mean(-1)
            squared = (target - output.point).square()
            normalized_squared = squared / scale.square()
            for i, record in enumerate(records):
                mask = valid[i]
                count = int(mask.sum())
                if not count:
                    raise ValueError("Evaluation table has no finite targets")
                loss_sum = float(pinball[i][mask].sum())
                sse = float(squared[i][mask].sum())
                nsse = float(normalized_squared[i][mask].sum())
                loss = loss_sum / count
                macro_loss.append(loss)
                points += count
                pinball_sum += loss_sum
                squared_sum += sse
                normalized_squared_sum += nsse
                table_writer.writerow(
                    {
                        "step": step,
                        "split": split,
                        "source": record["source"],
                        "points": count,
                        "pinball_loss": loss,
                        "rmse": math.sqrt(sse / count),
                        "normalized_rmse": math.sqrt(nsse / count),
                    }
                )
    finally:
        model.train(was_training)
    return {
        "pinball_loss": pinball_sum / points,
        "table_mean_pinball_loss": sum(macro_loss) / len(macro_loss),
        "rmse": math.sqrt(squared_sum / points),
        "normalized_rmse": math.sqrt(normalized_squared_sum / points),
        "points": points,
        "tables": len(macro_loss),
        "seconds": perf_counter() - started,
    }


def plot_history(history, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for split in ["train", "validation"]:
        axes[0].plot(
            [r["step"] for r in history],
            [r[split]["pinball_loss"] for r in history],
            label=split,
        )
        axes[1].plot(
            [r["step"] for r in history],
            [r[split]["normalized_rmse"] for r in history],
            label=split,
        )
    for ax, title in zip(
        axes[:2],
        ["Full-dataset normalized pinball loss", "Full-dataset normalized median RMSE"],
    ):
        ax.set(title=title, xlabel="Optimizer updates", yscale="log")
        ax.grid(alpha=0.25)
        ax.legend()
    lr_rows = [r for r in history if r.get("learning_rate") is not None]
    axes[2].plot([r["step"] for r in lr_rows], [r["learning_rate"] for r in lr_rows])
    axes[2].set(
        title="Learning rate used by each update",
        xlabel="Optimizer updates",
        ylabel="Learning rate",
    )
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)


@logged_training("capacity_test")
def run(config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    t = config["training"]
    directory = location(config, "training")
    directory.mkdir(parents=True, exist_ok=True)
    data = read_artifact(location(config, "embedded") / "embeddings.pt", config)
    if data["processed_sha256"] != file_digest(
        location(config, "processed") / "windows.pt"
    ):
        raise ValueError("Processed data changed; recreate embeddings")
    model = RWKVForecaster.load(ROOT / t["initial_checkpoint"]).train()
    if (
        signature(config)
        != read_artifact(ROOT / t["initial_checkpoint"], config)["signature"]
    ):
        raise ValueError("Initial model configuration mismatch")
    train_records = [r for r in data["records"] if r["split"] == "train"]
    validation_records = [r for r in data["records"] if r["split"] == "evaluation"]
    if not train_records or not validation_records:
        raise ValueError("Need both training and validation records")
    evaluation_batches = {
        "train": make_batches(train_records, t["evaluation_batch_size"]),
        "validation": make_batches(validation_records, t["evaluation_batch_size"]),
    }
    optimizer = MasterAdamW(model, t)
    steps_per_epoch = len(make_batches(train_records, t["batch_size"]))
    epoch_budget = t["epochs"] * steps_per_epoch
    total_steps = t["steps"] if t["steps"] is not None else epoch_budget
    if total_steps > epoch_budget:
        raise ValueError("training.steps exceeds the configured epoch budget")
    scheduler, schedule_plan = make_scheduler(optimizer.optimizer, t, total_steps)
    logger.info("Learning-rate schedule: {}", schedule_plan)
    write_json(
        directory / "learning_rate_scheduler.json",
        scheduler_snapshot(scheduler, schedule_plan),
    )
    report = architecture_report(model)
    write_json(directory / "architecture.json", report)
    emit_training_event(
        "model_initialized",
        parameters=report["total_parameters"],
        architecture=report,
        device=str(next(model.parameters()).device),
        dtype=str(next(model.parameters()).dtype),
    )
    emit_training_event(
        "training_plan",
        planned_steps=total_steps,
        configured_steps=t["steps"],
        train_tables=len(train_records),
        validation_tables=len(validation_records),
        training_schedule=schedule_plan,
        optimizer_precision="FP32 master weights and moments",
    )
    logger.info(
        "Capacity run: CUDA={} train_tables={} validation_tables={} epochs={} batch_size={} full evaluation EVERY update; FP32 master AdamW",
        torch.cuda.get_device_name(),
        len(train_records),
        len(validation_records),
        t["epochs"],
        t["batch_size"],
    )
    history = []
    step = 0
    target_points_processed = 0
    initial_projection = model.embedding.skip.weight.detach().float().clone()
    with (directory / "table_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as tables:
        writer = csv.DictWriter(
            tables,
            fieldnames=[
                "step",
                "split",
                "source",
                "points",
                "pinball_loss",
                "rmse",
                "normalized_rmse",
            ],
        )
        writer.writeheader()

        def sweep(epoch, update_loss=None, learning_rate=None):
            row = {
                "step": step,
                "epoch": epoch,
                "update_loss": update_loss,
                "learning_rate": learning_rate,
                "next_learning_rate": optimizer.optimizer.param_groups[0]["lr"],
            }
            for split, batches in evaluation_batches.items():
                row[split] = evaluate(model, batches, step, split, writer)
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            tables.flush()
            history.append(row)
            write_json(directory / "evaluation_history.json", history)
            logger.info("FULL EVALUATION {}", row)
            emit_training_event(
                "metrics",
                global_step=step,
                run_step=step,
                epoch=epoch,
                train_loss=row["train"]["pinball_loss"],
                batch_train_loss=update_loss,
                train_loss_scope="full_training_set",
                validation_loss=row["validation"]["pinball_loss"],
                validation_loss_scope="all_final_patches",
                learning_rate=learning_rate,
                next_learning_rate=row["next_learning_rate"],
                train_rmse=row["train"]["rmse"],
                validation_rmse=row["validation"]["rmse"],
                train_normalized_rmse=row["train"]["normalized_rmse"],
                validation_normalized_rmse=row["validation"]["normalized_rmse"],
                train_tables=row["train"]["tables"],
                validation_tables=row["validation"]["tables"],
                training_target_points_processed=target_points_processed,
            )

        sweep(0)
        for epoch in range(t["epochs"]):
            batches = make_batches(
                train_records, t["batch_size"], config["seed"] + epoch
            )
            for records in batches:
                if step >= total_steps:
                    break
                learning_rate = optimizer.optimizer.param_groups[0]["lr"]
                optimizer.zero_grad()
                prepared = on_device(pack_records(records))
                output = model.forward_encoded(model.encode_tokens(prepared), prepared)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise FloatingPointError("Invalid training loss")
                output.loss.backward()
                norm = optimizer.step(t["gradient_clip"])
                scheduler.step()
                step += 1
                target_points = int(torch.isfinite(prepared["labels"]).sum())
                target_points_processed += target_points
                emit_training_event(
                    "training_update",
                    global_step=step,
                    train_loss=float(output.loss.detach()),
                    learning_rate=learning_rate,
                    gradient_norm=norm,
                    target_points=target_points,
                    sources=[record["source"] for record in records],
                )
                logger.info(
                    "UPDATE step={} epoch={} batch_tables={} points={} loss={} grad_norm={} learning_rate={}",
                    step,
                    epoch + 1,
                    len(records),
                    int(torch.isfinite(prepared["labels"]).sum()),
                    float(output.loss.detach()),
                    norm,
                    learning_rate,
                )
                sweep(epoch + 1, float(output.loss.detach()), learning_rate)
            model.save(directory / "latest_weights.pt")
            write_json(
                directory / "learning_rate_scheduler.json",
                scheduler_snapshot(scheduler, schedule_plan),
            )
            plot_history(history, directory)
            if step >= total_steps:
                break
    save_artifact(
        directory / "model.pt",
        {
            "signature": signature(config),
            "config": data_config(model),
            "state_dict": model.state_dict(),
            "pipeline_config": config,
            "lr_scheduler": scheduler_snapshot(scheduler, schedule_plan),
            "embedding_sha256": file_digest(
                location(config, "embedded") / "embeddings.pt"
            ),
        },
    )
    summary = {
        "updates": step,
        "epochs": step // steps_per_epoch,
        "epoch_progress": step / steps_per_epoch,
        "planned_epochs": t["epochs"],
        "lr_scheduler": scheduler_snapshot(scheduler, schedule_plan),
        "initial": history[0],
        "final": history[-1],
        "best_validation": min(history, key=lambda r: r["validation"]["pinball_loss"]),
        "embedding_skip_weight_change_l2": float(
            (model.embedding.skip.weight.detach().float() - initial_projection).norm()
        ),
        "note": "All validation tables evaluated after every update. Repeatedly monitored final patches are validation, not an untouched test set. Training metrics are teacher-forced next-patch fit, not autonomous rollout.",
    }
    write_json(directory / "summary.json", summary)
    plot_history(history, directory)
    emit_training_event(
        "training_complete",
        global_step=step,
        completed_global_step=step,
        completed_run_step=step,
        planned_steps=total_steps,
        configured_steps=t["steps"],
        train_loss=history[-1]["train"]["pinball_loss"],
        validation_loss=history[-1]["validation"]["pinball_loss"],
        final_validation_loss=history[-1]["validation"]["pinball_loss"],
        full_training_corpus_pass_completed=step >= steps_per_epoch,
        training_target_points_processed=target_points_processed,
        checkpoint_path=str(directory / "model.pt"),
    )
    return summary


def data_config(model):
    from dataclasses import asdict

    return asdict(model.config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/capacity_test.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    with run_context(config, "capacity_test"):
        if launch_wsl(__file__, args.config):
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Capacity test requires CUDA")
        run(config)


if __name__ == "__main__":
    main()
