# Training restart blocked by the previous default checkpoint

On 2026-11-03, the prepared embedding manifest matches the current model and
data configuration: 22 layers, four direct output patches, a 64-point validation
tail. Training successfully read the prepared artifacts before its resume guard
rejected the old checkpoint. The existing `results/model/training/architecture.json`
describes the earlier 17-layer, 16-output-patch model. With `resume="auto"`,
the still-present `latest_checkpoint.pt` is selected for continuation.

Transform and embed regenerate preparation artifacts, not training checkpoints.
The corrective action is to archive the existing results directory and rerun
training. No additional transformation, embedding preparation, screening or raw
generation is needed for this specific failure:

```powershell
Move-Item -LiteralPath .\results\model -Destination ".\results\model_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
uv run python .\src\05_model_training\train.py
```

Keep `resume="auto"`: the empty new training directory causes fresh
initialization; future interruptions can resume the new compatible checkpoint.
No files were moved or deleted during diagnosis. The error message now identifies
the checkpoint and saved/requested layer counts and explicitly says that the
already-matching preparation stages need not be rerun. The previous error text
incorrectly prescribed regenerating artifacts even after they had passed their
configuration checks.
