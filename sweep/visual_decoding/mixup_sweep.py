import wandb

from scripts.visual_decoding.pipeline import ClassificationPipeline
from sweep.utils import create_sweep, save_best_run


def run_mixup_sweep(
    patient_dataset,
    loro_test_id,
    device,
    sweep_config_path,
    sweep_count=50,
    encoder_type="tcn",
    use_attention=False,
    adapter_dim=None,
):
    sweep_id = create_sweep(
        sweep_config_path, use_attention=use_attention, adapter_dim=adapter_dim
    )

    def train_fn():
        with wandb.init():
            config = dict(wandb.config)

            # Mixup training
            config["use_mixup"] = True
            if "mixup_layers" not in config:
                config["mixup_layers"] = [0, 1]

            scaler_mode = config["scaler_mode"]

            if adapter_dim is not None:
                if (
                    "adapter_dim" in config
                    and int(config["adapter_dim"]) != int(adapter_dim)
                ):
                    raise ValueError(
                        f"adapter_dim mismatch: sweep config has "
                        f"{config['adapter_dim']} but CLI passed {adapter_dim}"
                    )
                config["adapter_dim"] = adapter_dim
            elif "adapter_dim" not in config:
                config["adapter_dim"] = 0

            wandb.config.update(config, allow_val_change=True)

            pipeline = ClassificationPipeline(
                patient_dataset=patient_dataset,
                loro_test_id=loro_test_id,
                config=config,
                device=device,
                enable_wandb=True,
                encoder_type=encoder_type,
                freeze_encoder=False,
                use_attention=use_attention,
                scaler_mode=scaler_mode,
                attn_hidden_dim=config.get("attn_hidden_dim"),
            )

            pipeline.tune()

    wandb.agent(
        sweep_id,
        function=train_fn,
        count=sweep_count,
    )

    return sweep_id


def run_and_save_mixup_sweep(
    patient_dataset,
    loro_test_id,
    device,
    sweep_config_path,
    output_path,
    sweep_count=50,
    encoder_type="tcn",
    use_attention=False,
    adapter_dim=None,
):
    sweep_id = run_mixup_sweep(
        patient_dataset=patient_dataset,
        loro_test_id=loro_test_id,
        device=device,
        sweep_config_path=sweep_config_path,
        sweep_count=sweep_count,
        encoder_type=encoder_type,
        use_attention=use_attention,
        adapter_dim=adapter_dim,
    )

    best_config = save_best_run(
        sweep_id=sweep_id,
        metric="cv_val_loss",
        greater_is_better=False,
        output_path=output_path,
    )

    return best_config