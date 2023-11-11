import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from src.utils.cuda_status import get_num_gpus
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pathlib import Path
import time

logger = logging.getLogger(__name__)


def auto_accelerator(accelerator):
    if get_num_gpus() == 0 and accelerator == "gpu":
        accelerator = "cpu"
        logger.info("[auto_accelerator] gpu not detected and switch to cpu!")
    return accelerator


def setup_logger(dir, phase):
    name = "logs"
    version = "{}_{}".format(phase, time.strftime("%m%d-%H%M", time.localtime()))

    tb_logger = TensorBoardLogger(
        save_dir=dir,
        name=name,
        version=version + "_tflog",
    )

    csv_logger = CSVLogger(
        save_dir=dir,
        name=name,
        version=version + "_csvlog",
    )

    return [tb_logger, csv_logger]


def get_configure_callbacks(monitor, mode, patience, dir):
    # monitor = 'val_f1macro_epoch'
    # monitor = 'val_acc_epoch'
    earlystop = EarlyStopping(monitor=monitor, patience=patience, mode=mode)
    path = Path(dir).joinpath("model_checkpoint")
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    ckp_cb = ModelCheckpoint(
        dirpath=path,
        filename="bestCKP" + "-{epoch:02d}-{" + monitor + ":.3f}",
        monitor=monitor,
        save_top_k=1,
        mode=mode,
    )
    logger.info(f"[checkpoint] checkpoint file is stored in '{path}'")
    pb_cb = TQDMProgressBar(refresh_rate=0.2)

    return [earlystop, ckp_cb, pb_cb]


def get_trainer(args, phase):
    args.accelerator = auto_accelerator(args.accelerator)
    real_log_path = Path(args.log_dir).joinpath(args.expname).joinpath(args.midpath)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        max_epochs=args.epochs,
        logger=setup_logger(real_log_path, phase),
        callbacks=get_configure_callbacks(args.monitor, args.mode, args.patience, real_log_path),
    )
    return trainer
