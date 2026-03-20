import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from metrics.calculate_metrics import calculate_metrics
from utils.registry import MODEL_REGISTRY, ARCHITECTURE_REGISTRY, LOSS_REGISTRY

@MODEL_REGISTRY.register("AligNetModel")
class AligNetModel(pl.LightningModule):
    def __init__(self, config, val_names=None):
        super().__init__()
        self.save_hyperparameters()
        self.validation_names = val_names if val_names is not None else []
        self.automatic_optimization = False # Because we need to train first D and then G

        self.model_cfg = config["model"]
        self.data_cfg = config["datasets"]
        self.train_cfg = config.get("train", None)
        self.test_cfg = config.get("test", None)

        architecture_class = ARCHITECTURE_REGISTRY.get(self.model_cfg["architecture"])
        self.model = architecture_class(self.model_cfg)

        if self.train_cfg:
            freeze_policy = self.train_cfg.get("freeze_policy", None) 
            # "none" --> Nothing is frozen
            # "all" --> All layers are frozen
            # "encoder" --> Encoder is frozen
            # "except_encoder" --> All frozen except encoder
            # "except_encoder_decoder" --> All frozen except encoder and decoder
            if freeze_policy:
                if self.model_cfg.get("use_lora", False) and freeze_policy in ('all', 'encoder', 'encoder_parcial'):
                    print(f"[WARNING] When `use_lora=True`, `freeze_policy`cannot freeze the encoder, LoRA handles it automatically. Changing it to None.")
                    freeze_policy = None
                elif freeze_policy == "all":
                    print(f"[ALIGNetModel] All layers are forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        param.requires_grad = False
                elif freeze_policy == "except_encoder":
                    print(f"[ALIGNetModel] All layers except encoder are forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        if "encoder" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif freeze_policy == "except_encoder_decoder":
                    print(f"[ALIGNetModel] All layers except encoder and decoder are forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        if "encoder" in name or "decoder" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif freeze_policy == "encoder":
                    print(f"[ALIGNetModel] Encoder is forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        if "encoder" in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    self.model.encoder.eval()
                elif freeze_policy == "encoder_parcial":
                    print(f"[ALIGNetModel] Encoder is parcialy forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        param.requires_grad = False if "encoder" in name else True
                    for block in self.model.encoder.layer[-2:]:
                        for param in block.parameters():
                            param.requires_grad = True
                    for param in self.model.encoder.norm.parameters():
                        param.requires_grad = True
            # default: "none", nothing is frozen

        if self.train_cfg:
            loss_class = LOSS_REGISTRY.get(config["loss"]["type"])
            self.loss = loss_class(
                alpha_1=self.train_cfg["losses"]["alpha1"],
                alpha_2=self.train_cfg["losses"]["alpha2"],
                alpha_3=self.train_cfg["losses"]["alpha3"]
            )
            self.lr   = self.train_cfg["lr"]
            self.scheduler = self.train_cfg.get("scheduler", None)
            self.scheduler_gamma = self.train_cfg.get("scheduler_gamma", 0.99)

        if self.test_cfg:
            self.metric_results = {options['name']: {} for t, options in self.data_cfg.items()}

    def forward(self, img1, img2):
        pred_forward, pred_inverse = self.model(img1, img2)
        return pred_forward, pred_inverse

    def on_train_start(self):
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, (list, tuple)):
            for sch in schedulers:
                sch.step()  # aplica el primer step
        else:
            schedulers.step()

    def training_step(self, batch, batch_idx):
        img_r, img_s, _, _, gt_inverse, gt_forward, mask_r, mask_s = batch

        opt = self.optimizers()
        sch = self.lr_schedulers()

        pred_forward, pred_inverse = self(img_r, img_s)
        loss_reg, loss_tp, loss_nmi, loss_sym = self.loss(
            gt_inverse, gt_forward, pred_inverse, pred_forward, img_r, img_s, mask_r, mask_s, device=self.device
        )

        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss_reg)
        self.clip_gradients(
            opt,
            gradient_clip_val=0.5,
            gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
        )
        opt.step()
        if isinstance(sch, (list, tuple)):
            sch[0].step()
        else:
            sch.step()

        # Logging
        self.log('train/loss', loss_reg, prog_bar=True, sync_dist=True)
        self.log('train/loss_tp',  loss_tp,  prog_bar=True, sync_dist=True)
        self.log('train/loss_nmi', loss_nmi, sync_dist=True)
        self.log('train/loss_sym', loss_sym, sync_dist=True)

        # lr on bar (first optimizer shows)
        cur_lr = self.optimizers()[0].param_groups[0]['lr'] if isinstance(self.optimizers(), list) else self.optimizers().param_groups[0]['lr']
        self.log("lr", cur_lr, on_step=True, prog_bar=True)

        return loss_reg.detach()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img_r, img_s, _, _, gt_inverse, gt_forward, mask_r, mask_s = batch
        pred_forward, pred_inverse = self(img_r, img_s)

        loss_reg, loss_tp, loss_nmi, loss_sym = self.loss(
            gt_inverse, gt_forward, pred_inverse, pred_forward,
            img_r, img_s, mask_r, mask_s, device=self.device
        )

        log_dict = {
            f"{self.validation_names[dataloader_idx]}/loss": loss_reg,
            f"{self.validation_names[dataloader_idx]}/loss_tp": loss_tp,
            f"{self.validation_names[dataloader_idx]}/loss_nmi": loss_nmi,
            f"{self.validation_names[dataloader_idx]}/loss_sym": loss_sym,
        }

        self.log_dict(
            log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
            add_dataloader_idx=False
        )

        if dataloader_idx == 0:
            self.log("val_loss", loss_reg, on_epoch=True, sync_dist=True)

        # return loss_reg


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img_r, img_s, _, _, gt_inverse, gt_forward, mask_r, mask_s = batch
        pred_forward, pred_inverse = self(img_r, img_s)
        
        test_name = self.validation_names[dataloader_idx]
        self.metric_results[test_name] = calculate_metrics(img_r, img_s, gt_forward, gt_inverse, pred_forward, pred_inverse, mask_r, mask_s, 
                                                           self.test_cfg, self.metric_results[test_name], 
                                                           h=self.model_cfg["img_size"], w=self.model_cfg["img_size"])

    def on_test_epoch_end(self):
        test_dataloaders = self.trainer.test_dataloaders
        for idx, test_name in enumerate(self.validation_names):
            print(f"\nTest results for dataset: {test_name}:")
            num_samples = len(test_dataloaders[idx].dataset)
            for metric, value in self.metric_results[test_name].items():
                avg_value = value / (num_samples*2)     # x2 because we compute the metrics in both directions
                print(f"    * {metric}:    {avg_value:.5f}")
                
        self.metric_results = {name: {} for name in self.test_cfg['metrics'].keys()}

    def configure_optimizers(self):
        num_steps_epoch = self.trainer.num_training_batches
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"[AligNetModel] self.scheduler: {self.scheduler}", flush=True)

        if self.scheduler == "ExponentialLR":
            sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.scheduler_gamma)
            return {
                "optimizer": opt, 
                "lr_scheduler": {
                    "scheduler": sch, 
                    "interval": "epoch", 
                    "frequency": 1
                }
            }
        elif self.scheduler == "WarmUp_ExponentialLR":
            # Warmup for 3 epochs (linearly from lr/100 → lr)
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=3*num_steps_epoch)

            # Exponential decay afterwards
            decay = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.scheduler_gamma)

            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, decay], milestones=[3*num_steps_epoch])

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            raise Exception(f"LR scheduler '{self.scheduler}' is not implemented.")
        
        return opt