import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from lion_pytorch import Lion
from pt_utils import build_mask_like
from solitier_model import (
    SolitireEndToEndValueModel,
    SolitireMAEClassifyModel,
    SolitireMAEModel,
    SolitireValueModel,
)
from solitier_token import SPECIAL_TOKEN_BEGIN
from tqdm import tqdm

import wandb


def freeze_embeddings_and_rebuild_optimizer(
    model, optimizer, optimizer_type=Lion, lr=1e-4
):
    # 1) 勾配計算を止める
    model.embeddings.weight.requires_grad = False

    # 2) optimizer のパラメータから embeddings を除外して再構築
    #    再構築しないと weight decay がゼロ勾配でも掛かり続ける実装が多い
    #    `model.named_parameters()` で除外フィルタ
    params = [
        p for name, p in model.named_parameters() if p.requires_grad  # 勾配計算あり
    ]
    # 新しい optimizer を作る
    new_optimizer = optimizer_type(params, lr=lr)

    return new_optimizer


def mean_off_diag(embeddings: nn.Embedding, token_index_len: int) -> float:
    with torch.no_grad():
        E = embeddings.weight[:token_index_len]  # TokenOpen の範囲だけ
        E = torch.nn.functional.normalize(E, dim=1)
        sim = E @ E.T
        mean_offdiag = (sim.sum() - sim.diag().sum()) / (sim.numel() - len(E))
        # mean_offdiag が 0.3 以上に跳ねたら怪しい、などのしきい値を経験的に設定)
        return mean_offdiag.item()


class SolitireMAETrainer:
    def __init__(
        self,
        model: SolitireMAEModel,
        model_params: dict,
        lr: float = 1e-4,
        mask_ratio: float = 0.15,
        clip_grad_norm: float = 0.3,
        log_dir=r"/mnt/c/log/solitire/model/mae",
        eval_every_step: int = 0,
        max_steps: int = 1000000,
    ):
        self.model = model
        self.compiled_model = torch.compile(model, mode="reduce-overhead")
        self.model_params = model_params
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.clip_grad_norm = clip_grad_norm
        self.eval_every_step = eval_every_step
        model_params["mask_ratio"] = mask_ratio
        log_basename = "solitire_mae"
        log_basename += f"_m{model_params['dim']}"
        log_basename += f"_l{model_params['num_layer']}"
        log_basename += f"_h{model_params['num_heads']}"
        log_basename += f"_n{model_params['seq_len']}"
        log_basename += f"_ld{model_params['latent_dim']}"
        log_basename += f"_mr{model_params['mask_ratio']:0.2f}"
        if "balanced_sampler" in model_params and model_params["balanced_sampler"]:
            log_basename += f"_bs"
        if (
            "dir_stratified_sampler" in model_params
            and model_params["dir_stratified_sampler"]
        ):
            log_basename += f"_dss"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_basename += f"_{timestamp}"
        self.log_path = os.path.join(
            log_dir,
            log_basename,
        )
        os.makedirs(self.log_path, exist_ok=True)
        # wandb初期化
        wandb.init(
            project="solitire_mae",  # TensorBoardのlog_dirに相当する大きなカテゴリ
            name=log_basename,  # 各実験の名前
            config={  # ハイパーパラメータなどを渡すとwandbが自動記録
                "lr": self.lr,
                "mask_ratio": mask_ratio,
                "model_params": model_params,
                "clip_grad_norm": self.clip_grad_norm,
                "timestamp": timestamp,
            },
            tags=["solitire_mae", "solitire", "deep_solitire"],
        )
        self.device = torch.device("cuda")
        optimizer_type = Lion
        params = [
            p for name, p in model.named_parameters() if p.requires_grad  # 勾配計算あり
        ]
        self.optimizer = optimizer_type(params, lr=self.lr)
        warmup_steps = 3000
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: (s + 1) / warmup_steps
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr / 10.0
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, train_dataloader, val_dataloader=None):
        self.model.train()
        self.model.to(self.device)
        train_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {self.epoch}") as pbar:
            for batch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {"scheduler_lr": current_lr, "global_step": self.global_step},
                )
                indexes, _, _ = batch
                indexes = indexes.to(self.device)
                inputs = self.model.embeddings(indexes)
                normal_token_mask = indexes < SPECIAL_TOKEN_BEGIN
                normal_token_mask = normal_token_mask.to(self.device)
                mask = build_mask_like(indexes, self.mask_ratio)
                mask = mask.to(self.device)
                mask = mask & normal_token_mask

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.compiled_model.encode(inputs, mask=mask)
                    outputs = self.compiled_model.decode(latents)
                loss = self.criterion(outputs.float(), inputs)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                mean_off_diag_value = mean_off_diag(
                    self.model.embeddings, self.model_params["embeddings_len"]
                )
                wandb.log(
                    {
                        "mean_off_diag": mean_off_diag_value,
                        "step_Loss/train": loss.item(),
                        "global_step": self.global_step,
                    }
                )
                wandb.log(
                    {"Grad Norm/step": grad_norm, "global_step": self.global_step}
                )
                train_loss += loss.item()
                self.global_step += 1

                # ★Nステップ毎に in-epoch 検証（val_dataloader が与えられた場合のみ）
                if (val_dataloader is not None) and (self.eval_every_step > 0):
                    if self.global_step % self.eval_every_step == 0:
                        self._validate(val_dataloader, is_tqdm=False)

                        self.model.train()
                        self.model.to(self.device)

        avg_train_loss = train_loss / len(train_dataloader)
        wandb.log({"Loss/train": avg_train_loss, "epoch": self.epoch})
        return locals()

    def _validate(self, val_dataloader, is_tqdm=True):
        self.model.eval()
        self.model.to(self.device)

        val_loss = 0.0
        with torch.no_grad():
            if is_tqdm:
                pbar = tqdm(val_dataloader, desc=f"Validation Epoch {self.epoch}")
            else:
                pbar = val_dataloader
            for batch in pbar:
                indexes, _, _ = batch
                indexes = indexes.to(self.device)
                inputs = self.model.embeddings(indexes)
                normal_token_mask = indexes < SPECIAL_TOKEN_BEGIN
                normal_token_mask = normal_token_mask.to(self.device)
                mask = build_mask_like(indexes, self.mask_ratio)
                mask = mask.to(self.device)
                mask = mask & normal_token_mask

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.compiled_model.encode(inputs, mask=mask)
                    outputs = self.compiled_model.decode(latents)
                loss = self.criterion(outputs.float(), inputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({"step_Loss/val": avg_val_loss, "global_step": self.global_step})
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            self.save_model(prefix="best_")
        else:
            self.patience_counter += 1
        return avg_val_loss

    def validate_epoch(self, val_dataloader):
        avg_val_loss = self._validate(val_dataloader, is_tqdm=True)

        wandb.log({"Loss/val": avg_val_loss, "epoch": self.epoch})
        if not np.isnan(avg_val_loss):
            self.save_model(prefix="last_")
            self.save_model(prefix=f"epoch{self.epoch}_")
        self.epoch += 1
        return locals()

    def freeze_embeddings_and_rebuild_optimizer(self):
        self.optimizer = freeze_embeddings_and_rebuild_optimizer(
            self.model, self.optimizer, optimizer_type=Lion, lr=self.lr
        )
        print("Embeddings frozen and optimizer rebuilt.")

    def save_model(self, prefix=""):
        self.model.save_to_file(
            self.log_path,
            self.model_params,
            prefix=prefix,
        )


class SolitireMAEClassifyTrainer:
    def __init__(
        self,
        model: SolitireMAEClassifyModel,
        model_params: dict,
        lr: float = 1e-4,
        mask_ratio: float = 0.15,
        clip_grad_norm: float = 0.3,
        log_dir=r"/mnt/c/log/solitire/model/mae",
        eval_every_step: int = 0,
        max_steps: int = 1000000,
    ):
        self.model = model
        self.model_params = model_params
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.clip_grad_norm = clip_grad_norm
        self.eval_every_step = eval_every_step
        model_params["mask_ratio"] = mask_ratio
        log_basename = "solitire_mae_classify"
        log_basename += f"_m{model_params['dim']}"
        log_basename += f"_l{model_params['num_layer']}"
        log_basename += f"_h{model_params['num_heads']}"
        log_basename += f"_n{model_params['seq_len']}"
        log_basename += f"_ld{model_params['latent_dim']}"
        log_basename += f"_mr{model_params['mask_ratio']:0.2f}"
        if "balanced_sampler" in model_params and model_params["balanced_sampler"]:
            log_basename += f"_bs"
        if (
            "dir_stratified_sampler" in model_params
            and model_params["dir_stratified_sampler"]
        ):
            log_basename += f"_dss"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_basename += f"_{timestamp}"
        self.log_path = os.path.join(
            log_dir,
            log_basename,
        )
        os.makedirs(self.log_path, exist_ok=True)
        # wandb初期化
        wandb.init(
            project="solitire_mae",  # TensorBoardのlog_dirに相当する大きなカテゴリ
            name=log_basename,  # 各実験の名前
            config={  # ハイパーパラメータなどを渡すとwandbが自動記録
                "lr": self.lr,
                "mask_ratio": mask_ratio,
                "model_params": model_params,
                "clip_grad_norm": self.clip_grad_norm,
                "timestamp": timestamp,
            },
            tags=["solitire_mae", "solitire", "deep_solitire"],
        )
        self.device = torch.device("cuda")
        optimizer_type = Lion
        self.optimizer = optimizer_type(model.parameters(), lr=self.lr)
        warmup_steps = 3000
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: (s + 1) / warmup_steps
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr / 10.0
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        self.ignore_idx = -100
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_idx, label_smoothing=0.05
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, train_dataloader, val_dataloader=None):
        self.model.train()
        self.model.to(self.device)
        train_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {self.epoch}") as pbar:
            for batch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {"scheduler_lr": current_lr, "global_step": self.global_step},
                )
                indexes, _ = batch
                indexes = indexes.to(self.device)
                inputs = self.model.embeddings(indexes)
                normal_token_mask = indexes < SPECIAL_TOKEN_BEGIN
                normal_token_mask = normal_token_mask.to(self.device)
                mask = build_mask_like(indexes, self.mask_ratio)
                mask = mask.to(self.device)
                mask = mask & normal_token_mask

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.model.encode(inputs, mask=mask)
                    logits = self.model.decode(latents)
                # 損失は「通常トークン かつ マスク位置」のみ
                # それ以外（unmasked / 特殊トークン）は ignore_index に設定
                targets = indexes.clone()
                assert targets.dtype == torch.long
                loss_mask = mask & normal_token_mask  # 既存の2条件
                targets[~loss_mask] = self.ignore_idx

                # (B, L, C)→(B*L, C) と (B, L)→(B*L)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)).float(), targets.view(-1)
                )
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                wandb.log(
                    {
                        "step_Loss/train": loss.item(),
                        "global_step": self.global_step,
                    }
                )
                wandb.log(
                    {"Grad Norm/step": grad_norm, "global_step": self.global_step}
                )
                train_loss += loss.item()
                self.global_step += 1

                # ★Nステップ毎に in-epoch 検証（val_dataloader が与えられた場合のみ）
                if (val_dataloader is not None) and (self.eval_every_step > 0):
                    if self.global_step % self.eval_every_step == 0:
                        self._validate(val_dataloader, is_tqdm=False)

                        self.model.train()
                        self.model.to(self.device)

        avg_train_loss = train_loss / len(train_dataloader)
        wandb.log({"Loss/train": avg_train_loss, "epoch": self.epoch})
        return locals()

    def _validate(self, val_dataloader, is_tqdm=True):
        self.model.eval()
        self.model.to(self.device)

        val_loss = 0.0
        with torch.no_grad():
            if is_tqdm:
                pbar = tqdm(val_dataloader, desc=f"Validation Epoch {self.epoch}")
            else:
                pbar = val_dataloader
            for batch in pbar:
                indexes, _ = batch
                indexes = indexes.to(self.device)
                inputs = self.model.embeddings(indexes)
                normal_token_mask = indexes < SPECIAL_TOKEN_BEGIN
                normal_token_mask = normal_token_mask.to(self.device)
                mask = build_mask_like(indexes, self.mask_ratio)
                mask = mask.to(self.device)
                mask = mask & normal_token_mask

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.model.encode(inputs, mask=mask)
                    logits = self.model.decode(latents)
                # 損失は「通常トークン かつ マスク位置」のみ
                # それ以外（unmasked / 特殊トークン）は ignore_index に設定
                targets = indexes.clone()
                loss_mask = mask & normal_token_mask  # 既存の2条件
                targets[~loss_mask] = self.ignore_idx

                # (B, L, C)→(B*L, C) と (B, L)→(B*L)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)).float(), targets.view(-1)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({"step_Loss/val": avg_val_loss, "global_step": self.global_step})
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            self.save_model(prefix="best_")
        else:
            self.patience_counter += 1

    def validate_epoch(self, val_dataloader):
        avg_val_loss = self._validate(val_dataloader, is_tqdm=True)
        wandb.log({"Loss/val": avg_val_loss, "epoch": self.epoch})
        if not np.isnan(avg_val_loss):
            self.save_model(prefix="last_")
            self.save_model(prefix=f"epoch{self.epoch}_")
        self.epoch += 1
        return locals()

    def save_model(self, prefix=""):
        self.model.save_to_file(
            self.log_path,
            self.model_params,
            prefix=prefix,
        )


class SolitireValueTrainer:
    def __init__(
        self,
        model: SolitireValueModel,
        ae_model: SolitireMAEModel,
        model_params: dict,
        lr: float = 1e-4,
        clip_grad_norm: float = 0.3,
        log_dir=r"/mnt/c/log/solitire/model/value",
        eval_every_step: int = 0,
        max_steps: int = 1000000,
    ):
        self.model = model
        self.ae_model = ae_model
        self.model_params = model_params
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.eval_every_step = eval_every_step
        self.max_steps = max_steps
        log_basename = "solitire_value"
        log_basename += f"_m{model_params['dim']}"
        log_basename += f"_l{model_params['num_layer']}"
        log_basename += f"_h{model_params['num_heads']}"
        log_basename += f"_n{model_params['seq_len']}"
        log_basename += f"_cb{model_params['complete_bonus']:0.2f}"
        log_basename += f"_scr{model_params['source_complete_rate']:0.4f}"
        if "balanced_sampler" in model_params and model_params["balanced_sampler"]:
            log_basename += f"_bs"
        if (
            "dir_stratified_sampler" in model_params
            and model_params["dir_stratified_sampler"]
        ):
            log_basename += f"_dss"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_basename += f"_{timestamp}"
        self.log_path = os.path.join(
            log_dir,
            log_basename,
        )
        os.makedirs(self.log_path, exist_ok=True)
        # wandb初期化
        wandb.init(
            project="solitire_value",  # TensorBoardのlog_dirに相当する大きなカテゴリ
            name=log_basename,  # 各実験の名前
            config={  # ハイパーパラメータなどを渡すとwandbが自動記録
                "lr": self.lr,
                "model_params": model_params,
                "clip_grad_norm": self.clip_grad_norm,
                "timestamp": timestamp,
            },
            tags=["solitire_value", "solitire", "deep_solitire"],
        )
        self.device = torch.device("cuda")
        optimizer_type = Lion
        self.optimizer = optimizer_type(model.parameters(), lr=self.lr)
        warmup_steps = 3000
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: (s + 1) / warmup_steps
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr / 10.0
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, train_dataloader, val_dataloader=None):
        self.model.train()
        self.model.to(self.device)
        self.ae_model.eval()
        self.ae_model.to(self.device)
        train_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {self.epoch}") as pbar:
            for batch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {"scheduler_lr": current_lr, "global_step": self.global_step},
                )
                indexes, scores = batch
                scores = scores.to(self.device)
                indexes = indexes.to(self.device)
                inputs = self.ae_model.embeddings(indexes)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        latents = self.ae_model.encode(inputs)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(latents)
                loss = self.criterion(outputs.float(), scores)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                wandb.log(
                    {
                        "step_Loss/train": loss.item(),
                        "global_step": self.global_step,
                    }
                )
                wandb.log(
                    {"Grad Norm/step": grad_norm, "global_step": self.global_step}
                )
                train_loss += loss.item()
                self.global_step += 1

                # ★Nステップ毎に in-epoch 検証（val_dataloader が与えられた場合のみ）
                if (val_dataloader is not None) and (self.eval_every_step > 0):
                    if self.global_step % self.eval_every_step == 0:
                        self._validate(val_dataloader, is_tqdm=False)

                        self.model.train()
                        self.model.to(self.device)

        avg_train_loss = train_loss / len(train_dataloader)
        wandb.log({"Loss/train": avg_train_loss, "epoch": self.epoch})
        return locals()

    def _validate(self, val_dataloader, is_tqdm=True):
        self.model.eval()
        self.model.to(self.device)
        self.ae_model.eval()
        self.ae_model.to(self.device)

        val_loss = 0.0
        with torch.no_grad():
            if is_tqdm:
                pbar = tqdm(val_dataloader, desc=f"Validation Epoch {self.epoch}")
            else:
                pbar = val_dataloader
            for batch in pbar:
                indexes, scores = batch
                scores = scores.to(self.device)
                indexes = indexes.to(self.device)
                inputs = self.ae_model.embeddings(indexes)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latents = self.ae_model.encode(inputs)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(latents)
                loss = self.criterion(outputs.float(), scores)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({"step_Loss/val": avg_val_loss, "global_step": self.global_step})
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            self.save_model(prefix="best_")
        else:
            self.patience_counter += 1
        return avg_val_loss

    def validate_epoch(self, val_dataloader):
        avg_val_loss = self._validate(val_dataloader, is_tqdm=True)
        wandb.log({"Loss/val": avg_val_loss, "epoch": self.epoch})
        if not np.isnan(avg_val_loss):
            self.save_model(prefix="last_")
            self.save_model(prefix=f"epoch{self.epoch}_")
        self.epoch += 1
        return locals()

    def reduce_lr(self):
        """学習率を半分にする"""
        self.patience_counter = 0
        self.lr *= 0.5
        print("new lr: ", self.lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def save_model(self, prefix=""):
        self.model.save_to_file(
            self.log_path,
            self.model_params,
            prefix=prefix,
        )


class SolitireEndToEndValueTrainer:
    def __init__(
        self,
        model: SolitireEndToEndValueModel,
        model_params: dict,
        lr: float = 1e-4,
        clip_grad_norm: float = 0.3,
        log_dir=r"/mnt/c/log/solitire/model/value",
        eval_every_step: int = 0,
        max_steps: int = 1000000,
    ):
        self.model = model
        self.compiled_model = torch.compile(model)
        self.model_params = model_params
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.eval_every_step = eval_every_step
        log_basename = "solitire_endtoend_value"
        log_basename += f"_m{model_params['dim']}"
        log_basename += f"_l{model_params['num_layer']}"
        log_basename += f"_h{model_params['num_heads']}"
        log_basename += f"_n{model_params['seq_len']}"
        log_basename += f"_cb{model_params['complete_bonus']:0.2f}"
        log_basename += f"_scr{model_params['source_complete_rate']:0.4f}"
        if "balanced_sampler" in model_params and model_params["balanced_sampler"]:
            log_basename += f"_bs"
        if (
            "dir_stratified_sampler" in model_params
            and model_params["dir_stratified_sampler"]
        ):
            log_basename += f"_dss"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_basename += f"_{timestamp}"
        self.log_path = os.path.join(
            log_dir,
            log_basename,
        )
        os.makedirs(self.log_path, exist_ok=True)
        # wandb初期化
        wandb.init(
            project="solitire_value",  # TensorBoardのlog_dirに相当する大きなカテゴリ
            name=log_basename,  # 各実験の名前
            config={  # ハイパーパラメータなどを渡すとwandbが自動記録
                "lr": self.lr,
                "model_params": model_params,
                "clip_grad_norm": self.clip_grad_norm,
                "timestamp": timestamp,
            },
            tags=["solitire_value", "solitire", "deep_solitire"],
        )
        self.device = torch.device("cuda")
        optimizer_type = Lion
        self.optimizer = optimizer_type(model.parameters(), lr=self.lr)
        warmup_steps = 3000
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda s: (s + 1) / warmup_steps
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr / 10.0
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, train_dataloader, val_dataloader=None):
        """学習エポックを実行"""
        self.model.train()
        self.model.to(self.device)
        train_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {self.epoch}") as pbar:
            for batch in pbar:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {"scheduler_lr": current_lr, "global_step": self.global_step},
                )

                indexes, scores, _ = batch
                scores = scores.to(self.device)
                indexes = indexes.to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.compiled_model(indexes)
                loss = self.criterion(outputs.float(), scores)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                mean_off_diag_value = mean_off_diag(
                    self.model.embeddings, self.model_params["embeddings_len"]
                )
                wandb.log(
                    {
                        "mean_off_diag": mean_off_diag_value,
                        "step_Loss/train": loss.item(),
                        "global_step": self.global_step,
                    }
                )
                wandb.log(
                    {"Grad Norm/step": grad_norm, "global_step": self.global_step}
                )
                train_loss += loss.item()
                self.global_step += 1

                # ★Nステップ毎に in-epoch 検証（val_dataloader が与えられた場合のみ）
                if (val_dataloader is not None) and (self.eval_every_step > 0):
                    if self.global_step % self.eval_every_step == 0:
                        self._validate(val_dataloader, is_tqdm=False)

                        self.model.train()
                        self.model.to(self.device)

        avg_train_loss = train_loss / len(train_dataloader)
        wandb.log({"Loss/train": avg_train_loss, "epoch": self.epoch})
        return locals()

    def _validate(self, val_dataloader, is_tqdm=True):
        self.model.eval()
        self.model.to(self.device)

        val_loss = 0.0
        with torch.no_grad():
            if is_tqdm:
                pbar = tqdm(val_dataloader, desc=f"Validation Epoch {self.epoch}")
            else:
                pbar = val_dataloader
            for batch in pbar:
                indexes, scores, _ = batch
                scores = scores.to(self.device)
                indexes = indexes.to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.compiled_model(indexes)
                loss = self.criterion(outputs.float(), scores)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({"step_Loss/val": avg_val_loss, "global_step": self.global_step})
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            self.save_model(prefix="best_")
        else:
            self.patience_counter += 1
        return avg_val_loss

    def validate_epoch(self, val_dataloader):
        avg_val_loss = self._validate(val_dataloader)

        wandb.log({"Loss/val": avg_val_loss, "epoch": self.epoch})
        if not np.isnan(avg_val_loss):
            self.save_model(prefix="last_")
            self.save_model(prefix=f"epoch{self.epoch}_")
        self.epoch += 1

    def reduce_lr(self):
        """学習率を半分にする"""
        self.patience_counter = 0
        self.lr *= 0.5
        print("new lr: ", self.lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def save_model(self, prefix=""):
        self.model.save_to_file(
            self.log_path,
            self.model_params,
            prefix=prefix,
        )
