import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
from torch.utils.data import DataLoader
import yaml
import os
import glob

from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import TQDMProgressBar

import datasets
import architectures
import models
import losses
import metrics

from datasets.utils import create_dataloader, build_shared_cache
from utils.registry import DATASET_REGISTRY, MODEL_REGISTRY


torch.set_float32_matmul_precision('high')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="+", help="Override config keys, e.g. datasets.test_5.path=/new/path.csv")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.override:
        for ov in args.override:
            key, value = ov.split("=", 1)
            d = config
            parts = key.split(".")
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
    
    print("\n", yaml.dump(config, default_flow_style=False))

    seed = config["trainer"].get("seed", None)
    if seed:
        pl.seed_everything(seed, workers=True)
        print(f"\nSeed setted to {seed}.\n")
    
    # define image size if not defined
    config["model"]["img_size"] = config["model"].get("img_size", 224)

    # --- preload MDAS datasets into shared memory if present ---
    for test_key, test_cfg in config["datasets"].items():
        # only look for keys that start with "test"
        if not test_key.startswith("test"):
            continue

        ds_type = test_cfg.get("type", "")
        if ds_type in ("MDASDataset", "MDASCSVDataset"):
            print(f"[INFO] Building shared cache for {test_key} ({ds_type})")

            # instantiate dummy dataset (train=False for test phase)
            test_ds = DATASET_REGISTRY.get(ds_type)(
                root_path=test_cfg["path"],
                img_size=config["model"]["img_size"],
                train=False,
                cfg=test_cfg
            )

            # fill shared memory
            build_shared_cache(test_ds.file_paths, test_ds.resample_geotiff)

    # Create test dataloaders
    test_loaders = []
    test_names = []
    for key, test_cfg in config["datasets"].items():
        if key.startswith("test_"):
            test_loader = create_dataloader(
                test_cfg,
                config["model"]["img_size"],
                1,
                config["test"]["num_workers"],
                train=False
            )
            test_loaders.append(test_loader)
            test_names.append(test_cfg["name"])

    lengths_test = [len(dl.dataset) for dl in test_loaders]
    print(f"[DATA] {len(test_names)} test data loader/s created with {lengths_test} samples.\n", flush=True)

    trainer = pl.Trainer(
        gradient_clip_val=0.5,
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        enable_progress_bar=True,
    )

    model_class = MODEL_REGISTRY.get(config["model"]["model_name"])
    model = model_class(config, val_names=test_names)
    print(model, flush=True)

    ckpt = torch.load(config["test"]["checkpoint_path"], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"[TEST] Checkpoint in {config['test']['checkpoint_path']} loaded.")

    print(f"[TEST] Starting test...")
    trainer.test(model, test_loaders)
    