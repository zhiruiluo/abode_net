from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ml_toolkit.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.base_module.configs import ExpResults
from src.base_module.metrics_helper import Metrics_Helper, get_metrics
from src.config_options.modelbase_configs import HyperParm
from src.config_options.option_def import MyProgramArgs
from src.FlopsProfiler import FlopsProfiler
from src.utils.utils import get_datetime_now_tz

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True):
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    return total_params


class LightningBaseModule(pl.LightningModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__()
        self.save_hyperparameters(ignore=["args"])
        self.args = args
        self.baseconfig: HyperParm = self.args.modelBaseConfig
        self.metrics_init(self.args.modelConfig.nclass, self.baseconfig.label_mode)

    def metrics_init(self, nclass, label_mode):
        if label_mode == "multilabel":
            multilabel = True
            metric_keys = [
                "acc",
                "acc_perclass",
                "accmacro",
                "loss",
                "f1_perclass",
                "f1macro",
                "confmx",
            ]
        elif label_mode == "multiclass":
            multilabel = False
            metric_keys = [
                "acc",
                "accmacro",
                "loss",
                "f1macro",
                "f1micro",
                "f1none",
                "f1binary",
                "confmx",
            ]
        else:
            raise ValueError("label mode is invalid")

        self.all_metrics = nn.ModuleDict()

        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = nn.ModuleDict(
                get_metrics(metric_keys, nclass, multilabel=multilabel),
            )

        self._stored_confmx = {}

    def forward(self, batch):
        return batch

    def metrics(self, phase, pred, label, loss=None):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            if mk == "loss" and loss is not None:
                metric.update(loss)
            elif mk == "acc":
                metric(pred, label)
                self.log(
                    f"{phase}_acc_step",
                    metric,
                    sync_dist=True,
                    prog_bar=True,
                    batch_size=self.args.modelBaseConfig.batch_size,
                )
            else:
                metric.update(pred, label.to(torch.long))

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute().detach().cpu().tolist()
            metric.reset()
        # logger.info(metrics)
        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def get_all_confmx(self):
        return self._stored_confmx

    def log_epoch_end(self, phase, metrics):
        logger.info(f"Current Epoch: {self.current_epoch}")
        for k, v in metrics.items():
            if k == "confmx":
                logger.info(f'[{phase}_confmx] \n{metrics["confmx"]}')
                self._stored_confmx[f'{phase}_confmx'] = json.dumps(v)
                continue
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    self.log(f"{phase}_{k}_{i}", vi)
            else:
                self.log(f"{phase}_{k}", v)
            logger.info(f"[{phase}_{k}] {v}")

    def configure_optimizers(self):
        if self.args.modelBaseConfig.optimizer == 'AdamW':
            optimizer = AdamW(
                self.parameters(),
                lr=self.args.modelBaseConfig.lr,
                weight_decay=self.args.modelBaseConfig.weight_decay,
            )
        elif self.args.modelBaseConfig.optimizer == 'SGD':
            optimizer = SGD(
                self.parameters(),
                lr=self.args.modelBaseConfig.lr,
                weight_decay=self.args.modelBaseConfig.weight_decay,
            )
        else:
            raise ValueError('Optimizer is invalid')
            
        if self.args.modelBaseConfig.lr_scheduler == 'none':
            return {"optimizer": optimizer}
        elif self.args.modelBaseConfig.lr_scheduler == 'steplr':
            scheduler = StepLR(optimizer, step_size=7)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif self.args.modelBaseConfig.lr_scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=10)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            raise ValueError('Learning rate scheduler is invalid')

    def get_predict(self, y):
        a, y_hat = torch.max(y, dim=1)
        return y_hat

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        predictions = self.forward(batch)
        loss = predictions["loss"]
        output = predictions["output"]
        target = batch["target"]

        self.log(
            f"{phase}_loss_step",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.log(
            f"{phase}_loss_epoch",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
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
        target = batch["target"]
        predictions = self.forward(batch)
        output = predictions["output"]
        loss = predictions.get('loss', None)
        self.metrics(phase, output, target, loss)
        return

    def on_test_epoch_end(self) -> None:
        phase = "test"
        self.metrics_end(phase)


class LightningTrainerFactory:
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args

    def _get_logger(self, phase: str):
        name = f"logs"
        version = "{}_{}".format(phase, time.strftime("%m%d-%H%M", time.localtime()))

        tb_logger = TensorBoardLogger(
            save_dir=self.args.systemOption.task_dir,
            name=name,
            version=version+'_tflog',
        )

        csv_logger = CSVLogger(
            save_dir=self.args.systemOption.task_dir,
            name=name,
            version=version+'_csvlog',
        )

        return [tb_logger, csv_logger]

    def _configure_callbacks(self):
        callbacks = []
        monitor = self.args.trainerOption.monitor
        mode = self.args.trainerOption.mode

        earlystop = EarlyStopping(
            monitor=monitor,
            patience=self.args.modelBaseConfig.patience,
            mode=mode,
        )
        callbacks.append(earlystop)

        ckpt_path = Path(self.args.systemOption.task_dir).joinpath('ckpt')
        ckpt_path.mkdir(parents=True, exist_ok=True)
        ckp_cb = ModelCheckpoint(
            dirpath=ckpt_path,
            filename="bestCKP" + "-{epoch:02d}-" + "{" + monitor + ":.3f}",
            monitor=monitor,
            save_top_k=1,
            mode=mode,
        )
        callbacks.append(ckp_cb)

        if self.args.trainerOption.verbose:
            pb_cb = TQDMProgressBar(refresh_rate=0.05)
        else:
            pb_cb = TQDMProgressBar(refresh_rate=0)
        callbacks.append(pb_cb)

        lr_cb = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_cb)

        if self.args.nasOption.enable and self.args.nasOption.backend == "ray_tune":
            from ray.tune.integration.pytorch_lightning import \
                TuneReportCallback

            logger.info(f"[callbacks] ray_tune backend with TuneReportCallback")
            if self.baseconfig.label_mode == "multilabel":
                # metric_keys = ['acc_perclass', 'accmacro', 'f1_perclass', 'f1macro', 'confmx']
                metrics = {
                    "val_loss": "val_loss",
                    "val_acc": "val_acc",
                    "val_f1macro": "val_f1macro",
                    "val_epoch": "val_epoch",
                }
            elif self.baseconfig.label_mode == "multiclass":
                # metric_keys = ['acc','accmacro', 'loss', 'f1macro', 'f1micro', 'f1none','f1binary', 'confmx']
                metrics = {
                    "val_loss": "val_loss",
                    "val_acc": "val_acc",
                    "val_f1macro": "val_f1macro",
                    "val_epoch": "val_epoch",
                }
            else:
                raise ValueError("invalid label mode")

            callbacks.append(
                TuneReportCallback(metrics=metrics, on=["validation_epoch_end"]),
            )

        return callbacks

    def _auto_accelerator(self):
        accelerator = "cpu"
        if get_num_gpus() > 0 and not self.args.trainerOption.no_cuda:
            accelerator = "gpu"
        return accelerator

    def get_profiler(self):
        from pytorch_lightning.profilers import SimpleProfiler

        if self.args.trainerOption.profiler == "simple":
            return SimpleProfiler(
                dirpath=self.args.systemOption.task_dir,
                filename="profiler.txt",
            )

        return None

    def get_fit_trainer(self):
        self.args.trainerOption.accelerator = self._auto_accelerator()
        callbacks = [*self._configure_callbacks()]
        params = {
            "accelerator": self.args.trainerOption.accelerator,
            "devices": self.args.trainerOption.devices,
            "fast_dev_run": self.args.trainerOption.fast_dev_run,
            "precision": self.args.trainerOption.precision,
            "max_epochs": self.args.modelBaseConfig.epochs,
            # "auto_scale_batch_size": False
            # if self.args.trainerOption.auto_bs == ""
            # else self.args.trainerOption.auto_bs,
            "logger": self._get_logger("fit"),
            "callbacks": callbacks,
            "profiler": self.get_profiler(),
            "limit_train_batches": self.args.trainerOption.limit_train_batches,
            "limit_val_batches": self.args.trainerOption.limit_val_batches,
            "limit_test_batches": self.args.trainerOption.limit_test_batches,
        }
        logger.info(f"[fit_trainer] {params}")
        return pl.Trainer(**params)

    def get_val_trainer(self):
        accelerator = self._auto_accelerator()
        params = {
            "accelerator": accelerator,
            "devices": 1,
            "max_epochs": 1,
            "limit_val_batches": self.args.trainerOption.limit_val_batches,
            "logger": self._get_logger("val"),
        }
        
        if self.args.trainerOption.verbose:
            params['callbacks'] = [TQDMProgressBar(refresh_rate=0.05)]
        else:
            params['callbacks'] = [TQDMProgressBar(refresh_rate=0)]
        
        logger.info(f"[val_trainer] {params}")
        return pl.Trainer(**params)

    def get_test_trainer(self):
        accelerator = self._auto_accelerator()
        params = {
            "accelerator": accelerator,
            "devices": 1,
            "max_epochs": 1,
            "limit_test_batches": self.args.trainerOption.limit_test_batches,
            "logger": self._get_logger("test"),
        }
        
        if self.args.trainerOption.verbose:
            params['callbacks'] = [TQDMProgressBar(refresh_rate=0.05)]
        else:
            params['callbacks'] = [TQDMProgressBar(refresh_rate=0)]
        
        logger.info(f"[test_trainer] {params}")
        return pl.Trainer(**params)

    def training_flow(self, model: LightningBaseModule, dataset, no_test: bool = False) -> ExpResults:
        logger.info("[start training flow]")

        flops_profiler = FlopsProfiler(self.args)
        dataset.setup("fit")
        flops = None
        for batch in dataset.train_dataloader():
            if isinstance(batch, dict):
                new_batch = {}
                for k,v in batch.items():
                    new_batch[k] = v[0:1]
                flops = flops_profiler.get_flops(model, args=[new_batch])
            break

        fit_trainer = self.get_fit_trainer()
        time_on_fit_start = get_datetime_now_tz()
        fit_trainer.fit(model, datamodule=dataset)
        time_on_fit_end = get_datetime_now_tz()
        fit_results = fit_trainer.logged_metrics
        fit_results.update(model.get_all_confmx())

        ckp_cb = fit_trainer.checkpoint_callback
        earlystop_cb = fit_trainer.early_stopping_callback

        logger.info(
            "Interrupted %s, early stopped epoch %s",
            fit_trainer.interrupted,
            earlystop_cb.stopped_epoch,
        )
        # validation
        val_trainer = self.get_val_trainer()
        val_trainer.validate(
            model,
            ckpt_path=ckp_cb.best_model_path,
            datamodule=dataset,
        )
        val_results = val_trainer.logged_metrics
        val_results.update(model.get_all_confmx())

        # test model
        test_trainer = self.get_test_trainer()
        if (
            not no_test
            and os.path.isfile(ckp_cb.best_model_path)
            and not fit_trainer.interrupted
            and not val_trainer.interrupted
        ):
            time_on_test_start = get_datetime_now_tz()
            test_results = test_trainer.test(
                model,
                ckpt_path=ckp_cb.best_model_path,
                datamodule=dataset,
            )[0]
            time_on_test_end = get_datetime_now_tz()
            test_results.update(model.get_all_confmx())

        # delete check point
        if os.path.isfile(ckp_cb.best_model_path):
            os.remove(ckp_cb.best_model_path)

        # convert test_result dictionary to dictionary
        if (
            not fit_trainer.interrupted
            and not val_trainer.interrupted
            and not test_trainer.interrupted
        ):
            results = {}
            results.update(fit_results)
            results.update(val_results)
            if not no_test:
                results.update(test_results)
            results = Metrics_Helper.from_results(
                results,
                start_time=time_on_fit_start,
                training_time=time_on_fit_end - time_on_fit_start,
                flops=flops,
                params=self.args,
            )
            return results

        return None
