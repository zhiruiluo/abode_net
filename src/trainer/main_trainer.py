import sys
sys.path.append('.')
import logging
from pathlib import Path
import os

from src.models.model_select import ModelSelection
from src.datasets.dataset_select import DatasetSelection
from src.trainer.trainer_setting import get_trainer
from src.trainer.options import setup_arg
import pytorch_lightning as pl
import time
from src.logger.get_configured_logger import get_logger_seperate_config


logger = logging.getLogger(__name__)


def get_model(args, dl):
    args.nclass = dl.nclass
    from src.models.model_select import ModelSelection
    model = ModelSelection().getModel(args.model, args=args)
    return model

def lightning_train(args, dl, model, no_test: bool = False):
    logger.info(f'setup training environment')
    # setup pytorch-lightning trainer
    fit_trainer = get_trainer(args, 'fit')
    dl.setup('fit')
    # fit model
    fit_trainer.fit(model,datamodule=dl)
    fit_results = fit_trainer.logged_metrics
    fit_results.update(model.get_all_confmx())
    
    ckp_cb = fit_trainer.checkpoint_callback
    earlystop_cb = fit_trainer.early_stopping_callback

    logger.info(
            "Interrupted %s, early stopped epoch %s",
            fit_trainer.interrupted,
            earlystop_cb.stopped_epoch,
        )

    val_trainer = get_trainer(args, 'val')
    val_trainer.validate(
        model,
        ckpt_path=ckp_cb.best_model_path,
        datamodule=dl,
    )
    val_results = val_trainer.logged_metrics
    val_results.update(model.get_all_confmx())

    # test model
    dl.setup('test')
    test_trainer = get_trainer(args, 'test')
    if (
            not no_test
            and os.path.isfile(ckp_cb.best_model_path)
            and not fit_trainer.interrupted
            and not val_trainer.interrupted
        ):
        test_results = test_trainer.test(model, ckpt_path=ckp_cb.best_model_path, datamodule=dl)[0]
        test_results.update(model.get_all_confmx())
    
    # delete check point
    if args.delete_checkpoint and os.path.isfile(ckp_cb.best_model_path):
        os.remove(ckp_cb.best_model_path)
    
    ## convert test_result dictionary to dictionary
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
        logger.info('test results {}'.format(results))
    

def getlogger(args):
    jobid = os.environ.get('SLURM_JOB_ID')
    if jobid is not None:
        midpath = f'{args.model}_exp={args.exp}_{jobid}_ds={args.dataset}_{time.strftime("%m%d_%H%M", time.localtime())}/'
    else:
        midpath = f'{args.model}_ds={args.dataset}_{time.strftime("%m%d_%H%M", time.localtime())}/'
        
    fn = f"fold={args.fold}_t={time.strftime('%Y-%m-%d-%H:%M:%S')}_pid={os.getpid()}"
    root_logger = get_logger_seperate_config(args.debug, args.expname, midpath, fn)
    args.midpath = midpath
    return root_logger

def train():
    args = setup_arg()
    root_logger = getlogger(args)
    pl.seed_everything(args.seed)
    logger.info(f'[StartTraining] {args.dataset} {args.model}')
    dl = DatasetSelection.getDataset(args.dataset, args=args)
    # setup lighting model 
    model = get_model(args, dl)
    lightning_train(args, dl, model)