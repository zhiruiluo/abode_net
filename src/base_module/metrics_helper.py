from __future__ import annotations

import datetime
import logging
from typing import Any

import torchmetrics
from packaging.version import Version
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MeanMetric
from torchmetrics.classification import (
    BinaryF1Score,
    MultilabelAccuracy,
    MultilabelF1Score,
)

from src.base_module.configs import ExpResults, Metrics

logger = logging.getLogger(__name__)


def metric_builder_le_v09(name: str, nclass: int):
    dc = {
        "acc": Accuracy(),
        "accmacro": Accuracy(num_classes=nclass, average="macro"),
        "loss": MeanMetric(),
        "f1macro": F1Score(num_classes=nclass, average="macro"),
        "f1micro": F1Score(num_classes=nclass, average="micro"),
        "f1none": F1Score(num_classes=nclass, average="none"),
        "confmx": ConfusionMatrix(nclass),
    }
    print(name)
    metric = dc.get(name, None)
    if metric is None:
        raise ValueError(f"Metric {metric} is not defined")
    return metric


def metric_builder_ge_v010(name: str, nclass: int, multilabel: bool):
    if not multilabel:
        # if nclass == 2:
        #     task = 'binary'
        # else:
        task = "multiclass"

        dc = {
            "acc": Accuracy(task=task, num_classes=nclass),
            "accmacro": Accuracy(task=task, num_classes=nclass, average="macro"),
            "loss": MeanMetric(),
            "f1macro": F1Score(task="multiclass", num_classes=nclass, average="macro"),
            "f1micro": F1Score(task="multiclass", num_classes=nclass, average="micro"),
            "f1none": F1Score(task=task, num_classes=nclass, average="none"),
            "f1binary": BinaryF1Score(),
            "confmx": ConfusionMatrix(task=task, num_classes=nclass),
        }
    else:
        task = "multilabel"
        dc = {
            "acc": MultilabelAccuracy(num_labels=nclass, average="micro"),
            "acc_perclass": MultilabelAccuracy(num_labels=nclass, average="none"),
            "accmacro": MultilabelAccuracy(num_labels=nclass, average="macro"),
            "loss": MeanMetric(),
            "f1_perclass": MultilabelF1Score(num_labels=nclass, average="none"),
            "f1macro": MultilabelF1Score(num_labels=nclass, average="macro"),
            # "f1macro": F1Score(task=task, num_labels=nclass, average="macro"),
            "f1micro": F1Score(task=task, num_labels=nclass, average="micro"),
            "f1none": F1Score(task=task, num_labels=nclass, average="none"),
            "confmx": ConfusionMatrix(task=task, num_labels=nclass),
        }

    metric = dc.get(name, None)
    if metric is None:
        raise ValueError(f"Metric {metric} is not defined")
    return metric


def get_metrics(
    names: list[str] | str,
    nclass: int,
    multilabel: bool = False,
) -> dict[str,]:
    """
    >>> import torch
    >>> get_metrics('f1macro', 2)
    {'f1macro': BinaryF1Score()}

    >>> get_metrics('f1macro', 2)['f1macro'](torch.tensor([0,0,0,1,1,1]),torch.tensor([0,0,0,0,0,0]))
    tensor(0.)

    >>> get_metrics('f1macro', 3)
    {'f1macro': MulticlassF1Score()}

    >>> get_metrics('f1macro', 3)['f1macro'](torch.tensor([0,0,0,1,1,1]),torch.tensor([0,0,0,0,0,0]))
    tensor(0.2222)
    """
    logger.info(f"{names} {nclass}")
    torchmetrics_version = Version(torchmetrics.__version__)
    if torchmetrics_version <= Version("0.9.3"):
        builder = metric_builder_le_v09
        if multilabel:
            raise ValueError("This version torchmetrics doesn't support multilabel")
    else:
        builder = metric_builder_ge_v010

    if isinstance(names, str):
        names = [names]

    dc = {}

    for n in names:
        dc[n] = builder(n, nclass, multilabel)
    logger.info(dc)
    return dc


class Metrics_Helper:
    def __init__(self) -> None:
        ...

    @staticmethod
    def to_Metrics(results: dict[str, Any], phase: str) -> Metrics:
        res = {}
        for k in ["acc", "accmacro", "f1macro", "f1micro", "loss", "confmx"]:
            value = results.get(f"{phase}_{k}", None)
            if value is not None:
                if k == 'confmx':
                    value = str(value)
                else:
                    value = float(value)
            res[k] = value

        return Metrics(**res)

    @staticmethod
    def from_results(
        results: dict[str, Any],
        start_time: datetime.datetime,
        training_time: datetime.timedelta,
        macs: int = None,
        flops: int = None,
        params: dict = {},
    ) -> ExpResults:
        res = {}
        logger.info(results)
        for phase in ["train", "val", "test"]:
            metric = Metrics_Helper.to_Metrics(results, phase)
            res[f"{phase}_metrics"] = metric
        res["start_time"] = start_time
        res["training_time"] = training_time
        res["macs"] = macs
        res["flops"] = flops
        res["params"] = params
        logger.info(res)
        return ExpResults(**res)
