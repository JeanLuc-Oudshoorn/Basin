"""Where holdout results live: one timestamped folder per run, never overwritten."""

from datetime import datetime, timezone
from itertools import count

from transform import location

COMPLETED = "test_metrics.json"  # written last, so its presence marks a finished run


def holdout_directory(config):
    """Dedicated results beside the training directory, also for older configs."""
    if "holdout_test" in config["paths"]:
        return location(config, "holdout_test")
    return location(config, "training").parent / "holdout_test"


def new_holdout_run(config, *, epoch=None, step=None):
    """A new folder for one holdout run, and a config whose results go there.

    Every run gets its own timestamped folder under the holdout directory and
    nothing is ever written into an existing one, so earlier results (after
    epoch 1, epoch 2, ...) stay as they were.
    """
    root = holdout_directory(config)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if epoch is not None:
        name += f"_epoch{epoch}"
    if step is not None:
        name += f"_step{step}"
    for attempt in count(1):
        directory = root / (name if attempt == 1 else f"{name}_{attempt}")
        try:
            directory.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    paths = dict(config["paths"], holdout_test=str(directory))
    return directory, dict(config, paths=paths)


def resume_holdout_run(config, name):
    """The folder of an interrupted run, and a config that continues it.

    The stage then reads the tables that run saved and forecasts only the
    others. A finished run is never reopened.
    """
    directory = holdout_directory(config) / name
    if not directory.is_dir():
        raise FileNotFoundError(f"No holdout run to resume at {directory}")
    if (directory / COMPLETED).is_file():
        raise ValueError(f"{directory} is a finished run; there is nothing to resume")
    paths = dict(config["paths"], holdout_test=str(directory))
    holdout = dict(config.get("holdout", {}), resume=True)
    return directory, dict(config, paths=paths, holdout=holdout)


def holdout_runs(config):
    """Folders of completed runs, oldest first.

    Results saved before runs had their own folders lie directly in the
    holdout directory; that directory then counts as the oldest run.
    """
    root = holdout_directory(config)
    if not root.is_dir():
        return []
    runs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / COMPLETED).is_file()
    )
    return ([root] if (root / COMPLETED).is_file() else []) + runs


def latest_holdout_run(config, name=None):
    """The folder of the named run, or of the most recent completed one."""
    if name is not None:
        directory = holdout_directory(config) / name
        if not (directory / COMPLETED).is_file():
            raise FileNotFoundError(f"No completed holdout run at {directory}")
        return directory
    runs = holdout_runs(config)
    if not runs:
        raise FileNotFoundError(
            f"No completed holdout run under {holdout_directory(config)}"
        )
    return runs[-1]
