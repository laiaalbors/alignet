print("Starting to load the packages...", flush=True)

import wandb
import torch
import yaml
import os
import glob

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import TQDMProgressBar

import datasets
import architectures
import models
import losses
import metrics

from datasets.utils import create_dataloader, build_shared_cache
from utils.registry import MODEL_REGISTRY, DATASET_REGISTRY

print(f"All packages loaded.", flush=True)

torch.set_float32_matmul_precision('high')

class GradientNormLogger(pl.Callback):
    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every_n_steps = every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.every_n_steps == 0:
            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log using pl_module.log or just print
            pl_module.log('grad_norm', total_norm, on_step=True, prog_bar=True)

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        return None
    latest_ckpt = max(checkpoint_files, key=os.path.getctime)
    return latest_ckpt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print("\n", yaml.dump(config, default_flow_style=False), flush=True)

    seed = config["trainer"].get("seed", None)
    if seed:
        pl.seed_everything(seed, workers=True)
        print(f"\nSeed setted to {seed}.\n")

    # --- preload MDAS datasets into shared memory if present ---
    if config["datasets"]["train"]["type"] in ("MDASDataset", "MDASCSVDataset"):
        # make dummy instances to harvest paths + resample
        train_ds = DATASET_REGISTRY.get(config["datasets"]["train"]["type"])(root_path=config["datasets"]["train"]["path"], img_size=config["model"]["img_size"], train=True, cfg=config["datasets"]["train"])
        val_ds = DATASET_REGISTRY.get(config["datasets"]["validation_1"]["type"])(root_path=config["datasets"]["validation_1"]["path"], img_size=config["model"]["img_size"], train=True, cfg=config["datasets"]["validation_1"])
        # fill shared memory
        build_shared_cache(train_ds.file_paths, train_ds.resample_geotiff)
        build_shared_cache(val_ds.  file_paths,   val_ds.resample_geotiff)

    train_loader = create_dataloader(
        config["datasets"]["train"],
        config["model"]["img_size"],
        config["train"]["batch_size"],
        config["train"]["num_workers"]
    )
    print(f"[DATA] Train data loader created with {len(train_loader.dataset)} samples.", flush=True)

    # Create validation dataloaders
    val_loaders = []
    val_names = []
    for key, val_cfg in config["datasets"].items():
        if key.startswith("validation_"):
            val_loader = create_dataloader(
                val_cfg,
                config["model"]["img_size"],
                config["train"]["batch_size"]*2,
                config["train"]["num_workers"],
                train=False
            )
            val_loaders.append(val_loader)
            val_names.append(val_cfg["name"])

    lengths_val = [len(dl.dataset) for dl in val_loaders]
    print(f"[DATA] {len(val_names)} validation data loader/s created with {lengths_val} samples.\n", flush=True)

    wandb_logger = WandbLogger(
        project=config["wandb"]["project"],
        name=config["name"],
        id=config["wandb"].get("resume_id", None),
        resume="allow"
    )

    model_class = MODEL_REGISTRY.get(config["model"]["model_name"])
    model = model_class(config, val_names=val_names)
    print(model, flush=True)

    # 1) Best‐val model at end of epoch  
    val_ckpt = ModelCheckpoint(
        dirpath=os.path.join(config["train"]["checkpoint_dir"], config["name"]),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor=f"{val_names[0]}/loss_reg",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )

    # 2) Last checkpoint every N training steps  
    step_ckpt = ModelCheckpoint(
        dirpath=os.path.join(config["train"]["checkpoint_dir"], config["name"]),
        filename="last-step{step:06d}",
        save_last=True,
        every_n_train_steps=config["train"]["checkpoint_interval"],
        save_on_train_epoch_end=False,
        save_top_k=0
    )

    trainer = pl.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        strategy="ddp",
        precision=config["trainer"]["precision"],
        max_epochs=config["train"]["epochs"],
        logger=wandb_logger,
        callbacks=[val_ckpt, step_ckpt, GradientNormLogger(every_n_steps=100), TQDMProgressBar(refresh_rate=100)],
        val_check_interval=config["train"]["val_interval"],
        log_every_n_steps=config["train"]["log_interval"],
        enable_progress_bar=True,
        # strategy="ddp_find_unused_parameters_true",
    )

    if config["train"]["checkpoint_path"]:
        print(f"[TRAIN] Loading weights from {config['train']['checkpoint_path']}...", flush=True)
        state_dict = torch.load(config["train"]["checkpoint_path"], map_location='cpu', weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

        print("\n[VAL] Running validation before training...\n", flush=True)
        trainer.validate(model, val_loaders)
        
        trainer.fit(model, train_loader, val_loaders)
    
    else:
        last_ckpt_path = os.path.join(config["train"]["checkpoint_dir"], config["name"], "last.ckpt")
        
        if os.path.isfile(last_ckpt_path):
            ckpt_path = last_ckpt_path
        else:
            ckpt_path = find_latest_checkpoint(os.path.join(config["train"]["checkpoint_dir"], config["name"]))
        
        if ckpt_path:
            print(f"[TRAIN] Continuing training from checkpoint in {ckpt_path}...\n", flush=True)
        else:
            print("\n[VAL] Running validation before training...\n", flush=True)
            trainer.validate(model, val_loaders)

        trainer.fit(model, train_loader, val_loaders, ckpt_path=ckpt_path)
