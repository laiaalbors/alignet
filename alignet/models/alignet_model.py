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
        self.model = architecture_class(
            input_dim=self.model_cfg["input_channels"],
            num_heads=self.model_cfg["attention_heads"],
            window_size=self.model_cfg["window_size"],
            embed_dim=self.model_cfg["embed_dim"],
            encoder_dim_feedforward=self.model_cfg["encoder_dim_feedforward"],
            img_size=self.model_cfg["img_size"],
            num_transform_parameters=self.model_cfg["num_transform_parameters"], 
            shared_encoder=self.model_cfg.get("shared_encoder", True),
            shared_decoder=self.model_cfg.get("shared_decoder", True)
        )

        self.discriminator = None
        if self.model_cfg["discriminator"] is not None:
            discriminator_class = ARCHITECTURE_REGISTRY.get(self.model_cfg["discriminator"])
            self.discriminator = discriminator_class(
                img_size=self.model_cfg["img_size"],
                window_size=self.model_cfg["window_size"],
                embed_dim=self.model_cfg["embed_dim"],
                hidden=self.model_cfg["hidden_dis"],
                num_modalities=self.model_cfg["num_modalities"]
            )

        if self.train_cfg:
            freeze_policy = self.train_cfg.get("freeze_policy", None) 
            # "none" --> Nothing is frozen
            # "all" --> All layers are frozen, except the Discriminator
            # "except_encoder" --> All frozen except encoder and Discriminator
            # "except_encoder_decoder" --> All frozen except encoder and decoder and Discriminator
            if freeze_policy:
                if freeze_policy == "all":
                    print(f"[ALIGNetModel] All layers are forzen (except the Discriminator).", flush=True)
                    for name, param in self.model.named_parameters():
                        param.requires_grad = False
                elif freeze_policy == "except_encoder":
                    print(f"[ALIGNetModel] All layers except encoder (and Discriminator) are forzen.", flush=True)
                    for name, param in self.model.named_parameters():
                        if "encoder" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif freeze_policy == "except_encoder_decoder":
                    print(f"[ALIGNetModel] All layers except encoder and decoder (and Discriminator) are forzen.", flush=True)
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
            # default: "none", nothing is frozen

        if self.train_cfg:
            loss_class = LOSS_REGISTRY.get(config["loss"]["type"])
            self.loss_reg = loss_class(
                alpha_1=self.train_cfg["losses"]["alpha1"],
                alpha_2=self.train_cfg["losses"]["alpha2"],
                alpha_3=self.train_cfg["losses"]["alpha3"]
            )
            self.loss_adv = torch.nn.CrossEntropyLoss()
            self.loss_rec = torch.nn.L1Loss()

            self.lr_reg   = self.train_cfg["lr_reg"]
            self.lr_adv   = self.train_cfg.get("lr_adv", self.lr_reg)  # default
            
            self.lambda_adv = self.train_cfg["losses"].get("lambda_adv", 1.0)    # weight of adversarial loss
            self.use_grl = self.train_cfg.get("use_grl", True)         # toggle GRL vs uniform KL
            self.num_modalities = self.model_cfg.get("num_modalities", 2)

            self.scheduler = self.train_cfg.get("scheduler", None)
            self.scheduler_gamma = self.train_cfg.get("scheduler_gamma", 0.99)

        if self.test_cfg:
            self.metric_results = {options['name']: {} for t, options in self.data_cfg.items()}

    # Optional GRL (lightweight)
    @staticmethod
    def _grad_reverse(x, lambd=1.0):
        class _GR(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambd):
                ctx.lambd = lambd
                return x.view_as(x)
            @staticmethod
            def backward(ctx, g):
                return -ctx.lambd * g, None
        return _GR.apply(x, lambd)

    def _adv_lambda(self, global_step, max_steps):
        # Example: DANN-style sigmoid schedule
        p = global_step / max_steps
        return 0.5 * (2. / (1. + torch.exp(torch.tensor(-10 * p))) - 1.)

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
        img_r, img_s, modality1, modality2, gt_inverse, gt_forward, mask_r, mask_s = batch

        opt_g, opt_d = self.optimizers() if self.discriminator is not None else (self.optimizers(), None)
        # ^ When only one optimizer, self.optimizers() returns the single optimizer. But since we guard by self.discriminator, we're good.
        sch = self.lr_schedulers()

        # =========================
        # 1) Train Discriminator D
        # =========================
        if self.discriminator is not None:
            with torch.no_grad():
                emb_r = self.model.forward_encoder(img_r) #self.model.encode(img_r)
                emb_s = self.model.forward_encoder(img_s) #self.model.encode(img_s)
            feats = torch.cat([emb_r, emb_s], dim=0).detach()   # (2B, ...)
            labels = torch.cat([modality1, modality2], dim=0)     # (2B,)

            opt_d.zero_grad(set_to_none=True)
            logits_d = self.discriminator(feats)
            loss_d = self.loss_adv(logits_d, labels)
            self.manual_backward(loss_d)
            self.clip_gradients(
                opt_d,
                gradient_clip_val=0.5,
                gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
            )
            opt_d.step()
            if isinstance(sch, (list, tuple)):
                sch[1].step()

            self.log('train/loss_D', loss_d, prog_bar=True, sync_dist=True)

        # ======================
        # 2) Train Generator G
        # ======================
        pred_forward, pred_inverse = self(img_r, img_s)
        loss_reg, loss_tp, loss_nmi, loss_sym = self.loss_reg(
            gt_inverse, gt_forward, pred_inverse, pred_forward, img_r, img_s, mask_r, mask_s, device=self.device
        )

        loss_adv_g = torch.tensor(0.0, device=self.device)
        if self.discriminator is not None:
            # Reuse fresh embeddings (need gradients!)
            emb_r = self.model.forward_encoder(img_r) #self.model.encode(img_r)
            emb_s = self.model.forward_encoder(img_s) #self.model.encode(img_s)
            feats_g = torch.cat([emb_r, emb_s], dim=0)             # (2B, ...)
            labels  = torch.cat([modality1, modality2], dim=0)

            if self.use_grl:
                cur_lambda = self._adv_lambda(self.global_step, self.trainer.estimated_stepping_batches*self.trainer.num_training_batches//2)
                feats_g = self._grad_reverse(feats_g, cur_lambda)
                # feats_g = self._grad_reverse(feats_g, self.lambda_adv)
                logits_adv = self.discriminator(feats_g)
                loss_adv_g = self.loss_adv(logits_adv, labels)
            else:
                # Alternative: push to uniform (no GRL)
                logits = self.discriminator(feats_g)
                log_probs = F.log_softmax(logits, dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / self.num_modalities)
                loss_adv_g = F.kl_div(log_probs, uniform, reduction='batchmean')

        # loss_total = loss_reg + self.lambda_adv * loss_adv_g
        loss_total = loss_reg + 5*loss_adv_g

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_total)
        self.clip_gradients(
            opt_g,
            gradient_clip_val=0.5,
            gradient_clip_algorithm=getattr(self.trainer, "gradient_clip_algorithm", "norm"),
        )
        opt_g.step()
        if isinstance(sch, (list, tuple)):
            sch[0].step()
        else:
            sch.step()

        # Logging
        self.log('train/loss_reg', loss_reg, prog_bar=True, sync_dist=True)
        self.log('train/loss_tp',  loss_tp,  prog_bar=True, sync_dist=True)
        self.log('train/loss_nmi', loss_nmi, sync_dist=True)
        self.log('train/loss_sym', loss_sym, sync_dist=True)
        if self.discriminator is not None:
            self.log('train/loss_adv_G', loss_adv_g, prog_bar=True, sync_dist=True)
            self.log('train/loss_total', loss_total, prog_bar=True, sync_dist=True)
            self.log("train/lambda_adv", torch.tensor(cur_lambda, device=self.device), prog_bar=True, on_step=True, sync_dist=True)

        # lr on bar (first optimizer shows)
        cur_lr = self.optimizers()[0].param_groups[0]['lr'] if isinstance(self.optimizers(), list) else self.optimizers().param_groups[0]['lr']
        self.log("lr", cur_lr, on_step=True, prog_bar=True)

        return loss_total.detach()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img_r, img_s, modality1, modality2, gt_inverse, gt_forward, mask_r, mask_s = batch
        pred_forward, pred_inverse = self(img_r, img_s)

        # --- registration loss ---
        loss_reg, loss_tp, loss_nmi, loss_sym = self.loss_reg(
            gt_inverse, gt_forward, pred_inverse, pred_forward,
            img_r, img_s, mask_r, mask_s, device=self.device
        )

        log_dict = {
            f"{self.validation_names[dataloader_idx]}/loss_reg": loss_reg,
            f"{self.validation_names[dataloader_idx]}/loss_tp": loss_tp,
            f"{self.validation_names[dataloader_idx]}/loss_nmi": loss_nmi,
            f"{self.validation_names[dataloader_idx]}/loss_sym": loss_sym,
        }

        # --- discriminator + adversarial loss (only if discriminator exists) ---
        loss_adv_g = torch.tensor(0.0, device=self.device)
        if self.discriminator is not None:
            with torch.no_grad():
                emb_r = self.model.forward_encoder(img_r) #self.model.encode(img_r)
                emb_s = self.model.forward_encoder(img_s) #self.model.encode(img_s)

            feats = torch.cat([emb_r, emb_s], dim=0)
            labels = torch.cat([modality1, modality2], dim=0)

            # Discriminator loss (features detached in training, here no grad anyway)
            logits_d = self.discriminator(feats)
            loss_d = self.loss_adv(logits_d, labels)

            # Compute Accuracy of the Discriminator
            preds = torch.argmax(logits_d, dim=1)        # clase predicha
            correct = (preds == labels).sum().item()     # nº de aciertos
            total = labels.size(0)                       # nº de ejemplos
            accuracy_d = correct / total

            # Generator adversarial loss
            if self.use_grl:
                cur_lambda = self._adv_lambda(self.global_step, self.trainer.estimated_stepping_batches*self.trainer.num_training_batches//2)
                feats_g = self._grad_reverse(feats, cur_lambda)
                # feats_g = self._grad_reverse(feats, self.lambda_adv)
                logits_adv = self.discriminator(feats_g)
                loss_adv_g = self.loss_adv(logits_adv, labels)
            else:
                logits = self.discriminator(feats)
                log_probs = F.log_softmax(logits, dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / self.num_modalities)
                loss_adv_g = F.kl_div(log_probs, uniform, reduction='batchmean')

            log_dict.update({
                f"{self.validation_names[dataloader_idx]}/loss_D": loss_d,
                f"{self.validation_names[dataloader_idx]}/loss_adv_G": loss_adv_g,
                f"{self.validation_names[dataloader_idx]}/accuracy_D": accuracy_d,
            })

        # loss_total = loss_reg + self.lambda_adv * loss_adv_g
        loss_total = loss_reg + 5*loss_adv_g

        log_dict.update({
            f"{self.validation_names[dataloader_idx]}/loss_total": loss_total,
        })

        self.log_dict(
            log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
            add_dataloader_idx=False
        )

        final_val_loss = loss_total if self.discriminator is not None else loss_reg
        return final_val_loss


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img_r, img_s, modality1, modality2, gt_inverse, gt_forward, mask_r, mask_s = batch
        pred_forward, pred_inverse = self(img_r, img_s)
        
        test_name = self.validation_names[dataloader_idx]
        self.metric_results[test_name] = calculate_metrics(img_r, img_s, gt_forward, gt_inverse, pred_forward, pred_inverse, mask_r, mask_s, 
                                                           self.test_cfg, self.metric_results[test_name], 
                                                           h=self.model_cfg["img_size"], w=self.model_cfg["img_size"])

        if self.discriminator is not None:
            with torch.no_grad():
                emb_r = self.model.forward_encoder(img_r) #self.model.encode(img_r)
                emb_s = self.model.forward_encoder(img_s) #self.model.encode(img_s)
            feats = torch.cat([emb_r, emb_s], dim=0).detach()   # (2B, ...)
            labels = torch.cat([modality1, modality2], dim=0)     # (2B,)
            logits_d = self.discriminator(feats)
            preds = torch.argmax(logits_d, dim=1)        # clase predicha
            correct = (preds == labels).sum().item()     # nº de aciertos
            total = labels.size(0)                       # nº de ejemplos
            accuracy = correct / total
            if 'accuracy_D' not in self.metric_results[test_name]:
                self.metric_results[test_name]['accuracy_D'] = 0
            self.metric_results[test_name]['accuracy_D'] += accuracy

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
        print(f"[AligNetModel] self.scheduler: {self.scheduler}", flush=True)

        # Return two optimizers (G first, D second) for clarity
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_reg)
        if self.discriminator is not None:
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_adv)
            # Optional schedulers:
            if self.scheduler == "ExponentialLR":
                sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.scheduler_gamma)
                sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.scheduler_gamma)
                return [opt_g, opt_d], [
                    {"scheduler": sch_g, "interval": "epoch", "frequency": 1},
                    {"scheduler": sch_d, "interval": "epoch", "frequency": 1},
                ]
            elif self.scheduler == "WarmUp_ExponentialLR":
                sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.scheduler_gamma)

                # Warmup for 3 epochs (linearly from lr_reg/100 → lr_reg)
                warmup = torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.01, end_factor=1.0, total_iters=3*14229)

                # Exponential decay afterwards
                decay = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.scheduler_gamma)

                scheduler = torch.optim.lr_scheduler.SequentialLR(opt_g, schedulers=[warmup, decay], milestones=[3*14229])

                return [opt_g, opt_d], [
                    {"scheduler": scheduler, "interval": "step", "frequency": 1},
                    {"scheduler": sch_d, "interval": "epoch", "frequency": 1},
                ]
            
            return [opt_g, opt_d]
        else:
            if self.scheduler == "ExponentialLR":
                sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.scheduler_gamma)
                return {"optimizer": opt_g, "lr_scheduler": {"scheduler": sch_g, "interval": "epoch", "frequency": 1}}
            elif self.scheduler == "WarmUp_ExponentialLR":
                # Warmup for 3 epochs (linearly from lr_reg/100 → lr_reg)
                warmup = torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.01, end_factor=1.0, total_iters=3*14229)

                # Exponential decay afterwards
                decay = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.scheduler_gamma)

                scheduler = torch.optim.lr_scheduler.SequentialLR(opt_g, schedulers=[warmup, decay], milestones=[3*14229])

                return {
                    "optimizer": opt_g,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1
                    }
                }
            return opt_g