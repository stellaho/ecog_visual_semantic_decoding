import json
import logging
import math
from pathlib import Path

import wandb


SWEEP_PROJECT = "ecog_visuals"

SUMMARY_KEYS = (
    "cv_val_loss",
    "fold_best_val_losses",
    "fold_best_epochs",
    "final_fit_epochs",
    "fold_episode_sizes",
)


def create_sweep(sweep_config_path, project=None):
    project = project or SWEEP_PROJECT

    config = json.loads(
        Path(sweep_config_path).read_text()
    )

    return wandb.sweep(
        config,
        project=project,
    )


def save_best_run(
    sweep_id,
    metric,
    greater_is_better=False,
    output_path=None,
    project=None,
):
    project = project or SWEEP_PROJECT

    sweep = wandb.Api().sweep(
        f"{project}/{sweep_id}"
    )

    runs = []

    for run in sweep.runs:
        if run.state != "finished":
            continue

        value = run.summary.get(metric)

        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue

        runs.append(run)

    if not runs:
        logging.getLogger(__name__).warning(
            "No finished runs found with metric '%s'",
            metric,
        )
        return None

    best_run = (
        max(runs, key=lambda run: run.summary[metric])
        if greater_is_better
        else min(runs, key=lambda run: run.summary[metric])
    )

    best_config = {
        key: value
        for key, value in best_run.config.items()
        if not key.startswith("_")
    }

    for key in SUMMARY_KEYS:
        if key in best_run.summary:
            best_config[key] = best_run.summary[key]

    if output_path is None:
        output_path = (
            Path("./config/visual_decoding/sweeps/saved")
            / f"{sweep_id}.json"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(best_config, indent=2) + "\n"
    )

    return best_config