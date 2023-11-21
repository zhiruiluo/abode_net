import argparse
import logging

logger = logging.getLogger(__name__)

from src.models.model_select import ModelSelection
from src.datasets.dataset_select import DatasetSelection


def add_model_hyperparameter(parser, model):
    # model specific hyperparameters are returned from the get_options function located in the corresponding model file.
    # format [param_name, type, default] e.g., ['nclass', int, 2,] => param_name='nclass', type=int, default='2'
    _, options = ModelSelection.getParams(model)
    for o in options:
        parser.add_argument("--" + o[0], type=o[1], default=o[2])


def add_dataset_parameter(parser, dataset):
    # dataset specific hyperparameters are returned from the get_options function located in the corresponding dataset file.
    # format [param_name, type, default] e.g., ['nclass', int, 2,] => param_name='nclass', type=int, default='2'
    _, options = DatasetSelection.getParams(dataset)
    for o in options:
        parser.add_argument("--" + o[0], type=o[1], default=o[2])


def add_trainer_parameter(parser):
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--patience", type=int, default=12, help="Patience for early stopping.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument("--determ", action="store_true", help="Deterministic flag")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--profiler", action="store_true")
    # parser.add_argument('--label_smoothing', type=float, default=0.0, help='0 <= label smoothing < 1, 0 no smoothing')
    parser.add_argument("--monitor", type=str, default="val_f1macro")
    parser.add_argument("--mode", type=str, default="max")
    parser.add_argument("--log_dir", type=str, default="logging")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--delete_checkpoint", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers used in dataloader",
    )


def add_experiment_parameter(parser):
    parser.add_argument(
        "--model",
        type=str,
        default=ModelSelection.default_model(),
        choices=ModelSelection.choices(),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DatasetSelection.default(),
        help="dataset choice",
        choices=DatasetSelection.choices(),
    )
    parser.add_argument("--exp", type=int, default=1, help="Unique experiment number")
    parser.add_argument("--nfold", type=int, default=1)
    parser.add_argument("--nrepeat", type=int, default=1)


def add_results_parameter(parser):
    parser.add_argument("--fold", type=int, default=1, help="The fold number for current training")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--splits", type=str, default="3:1:1")


def add_system_parameter(parser):
    parser.add_argument("--expname", type=str, default="testexp")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("--seed", type=int, default=42)


def pre_build_arg():
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument("--dataset", type=str, default=DatasetSelection.default())
    model_parser.add_argument("--model", type=str, default=ModelSelection.default_model())
    dataset = model_parser.parse_known_args()[0].dataset
    model = model_parser.parse_known_args()[0].model
    return dataset, model


def build_arg(dataset, model):
    parser = argparse.ArgumentParser()
    add_system_parameter(parser)
    add_dataset_parameter(parser, dataset)
    add_model_hyperparameter(parser, model)
    add_experiment_parameter(parser)
    add_results_parameter(parser)
    add_trainer_parameter(parser)
    return parser


def setup_arg():
    dataset, model = pre_build_arg()
    parser = build_arg(dataset, model)
    return parser.parse_args()
