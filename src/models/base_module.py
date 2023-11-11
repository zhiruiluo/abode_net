import logging
import pytorch_lightning as pl
import pandas as pd
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from src.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim import SGD, AdamW, Adam
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from src.models.metrics_helper import get_metrics
import time

logger = logging.getLogger(__name__)


def pretty_print_confmx(confmx):
    return (
        " ".join([f"r{i:<3d}" for i in range(len(confmx))])
        + f" sum"
        + "\n"
        + "\n".join(
            [" ".join([f"{cell.item():4d}" for cell in row]) + f"{sum(row):4d}" for row in confmx]
        )
    )


def pretty_print_confmx_pandas(confmx):
    pd.set_option("display.max_columns", None)
    df_confmx = pd.DataFrame(confmx.numpy())
    df_confmx["sum"] = df_confmx.sum(axis=1)
    str_confmx = str(df_confmx)
    pd.reset_option("display.max_columns")
    return str_confmx


class LightningBaseModule(pl.LightningModule):
    def __init__(self, nclass: int, optimizer: str, lr_scheduler: str):
        super(LightningBaseModule, self).__init__()
        self.save_hyperparameters()
        self.nclass = nclass
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics_init(nclass)

    def metrics_init(self, nclass: int):
        metric_keys = [
            "acc",
            "accmacro",
            "loss",
            "f1macro",
            "f1micro",
            "f1none",
            "confmx",
        ]
        self.all_metrics = nn.ModuleDict()
        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = nn.ModuleDict(
                get_metrics(metric_keys, nclass, multilabel=False)
            )

        self._stored_confmx = {}

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def metrics(self, phase, pred, label, loss=None):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            if mk == "loss" and loss is not None:
                result = metric.update(loss)
            elif mk == "acc":
                result = metric(pred, label)
                self.log(
                    f"{phase}_acc_step",
                    result,
                    sync_dist=True,
                    prog_bar=True,
                    batch_size=self.args.batch_size,
                )
            else:
                metric.update(pred, label.to(torch.long))

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute().detach().cpu().tolist()
            metric.reset()

        self.log_epoch_end(phase, metrics)

    def get_all_confmx(self):
        return self._stored_confmx

    def log_epoch_end(self, phase, metrics):
        logger.info(f"Current Epoch: {self.current_epoch}")
        for k, v in metrics.items():
            if k == "confmx":
                logger.info(f'[{phase}_confmx] \n{metrics["confmx"]}')
                self._stored_confmx[f"{phase}_confmx"] = json.dumps(v)
                continue
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    self.log(f"{phase}_{k}_{i}", vi)
            else:
                self.log(f"{phase}_{k}", v)
            logger.info(f"[{phase}_{k}] {v}")

    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.optimizer == "Adam":
            optimizer = Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError("Optimizer is invalid")

        if self.args.lr_scheduler == "none":
            return {"optimizer": optimizer}
        elif self.args.lr_scheduler == "steplr":
            scheduler = StepLR(optimizer, step_size=7)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.lr_scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=7, num_training_steps=self.args.epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.lr_scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=7, num_training_steps=self.args.epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError("Learning rate scheduler is invalid")

    def shared_my_step(self, batch, batch_nb, phase):
        predictions = self.forward(batch)
        loss = predictions["loss"]
        output = predictions["output"]
        target = batch["target"]

        self.log(
            f"{phase}_loss_step",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )
        self.log(
            f"{phase}_loss_epoch",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.args.batch_size,
        )

        self.metrics(phase, output, target, loss)
        return loss

    def training_step(self, batch, batch_nb):
        phase = "train"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def on_train_epoch_end(self) -> None:
        phase = "train"
        self.metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = "val"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def on_validation_epoch_end(self) -> None:
        phase = "val"
        self.metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = "test"
        predictions = self.forward(batch)
        loss = predictions["loss"]
        output = predictions["output"]
        target = batch["target"]

        self.metrics(phase, output, target, loss)
        return

    def on_test_epoch_end(self) -> None:
        phase = "test"
        self.metrics_end(phase)
