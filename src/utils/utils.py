from __future__ import annotations

import datetime
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def get_datetime_now() -> datetime.datetime:
    return datetime.datetime.now()


def get_datetime_now_tz() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).astimezone()


def get_num_gpus():
    import torch

    num_gpus = torch.cuda.device_count()
    logger.info(f"[num_gpus] {num_gpus}")
    return num_gpus


def cpu_count() -> int:
    if os.environ.get("SLURM_CPUS_ON_NODE"):
        cpus_reserved = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        cpus_reserved = 8
    return cpus_reserved


def cuda_available() -> bool:
    if get_num_gpus() == 0:
        return False
    return True


def set_cpu_affinity(num_cpus) -> None:
    import numpy as np

    logger.info("[cpu affinity] %s", os.sched_getaffinity(0))
    cpu_count = os.cpu_count()
    choices = np.random.choice(
        list(range(1, cpu_count)),
        replace=False,
        size=num_cpus,
    ).tolist()
    if choices == []:
        logger.info("[cpu affinity] use default setting %s", os.sched_getaffinity(0))
        return
    os.sched_setaffinity(0, choices)
    logger.info("[cpu affinity] set new affinity %s", os.sched_getaffinity(0))


def pretty_print_confmx_pandas(confmx):
    import pandas as pd

    if not isinstance(confmx, list) and not isinstance(confmx, np.ndarray):
        confmx = confmx.numpy()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 500)
    df_confmx = pd.DataFrame(confmx)
    df_confmx["sum"] = df_confmx.sum(axis=1)
    str_confmx = str(df_confmx)
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    return str_confmx


def get_padding_same(kernel: int | tuple, dilation: int | tuple = 1, num_dim: int = 2):
    """Calculate the padding for `same` in different strides

    >>> get_padding_same(3, 1)
    (1, 1)

    >>> get_padding_same(5, 1)
    (2, 2)

    >>> get_padding_same(4, 1)
    (1, 1)

    >>> get_padding_same((3,5), 1)
    (1, 2)
    """
    kernels = None
    if isinstance(kernel, int):
        kernels = (kernel,) * num_dim
    else:
        kernels = (k for k in kernel)

    dilations = None
    if isinstance(dilation, int):
        dilations = (dilation,) * num_dim
    else:
        dilations = (d for d in dilation)

    paddings = ()
    for d, k in zip(dilations, kernels):
        paddings += (d * (k - 1) // 2,)

    return paddings


def get_padding_one_more_or_same(
    kernel: int | tuple,
    dilation: int | tuple = 1,
    num_dim: int = 2,
):
    """Calculate the padding for `same` in different strides

    >>> get_padding_one_more_or_same(3, 1)
    (1, 1)

    >>> get_padding_one_more_or_same(5, 1)
    (2, 2)

    >>> get_padding_one_more_or_same(4, 1)
    (2, 2)

    >>> get_padding_one_more_or_same((3,5), 1)
    (1, 2)
    """
    kernels = None
    if isinstance(kernel, int):
        kernels = (kernel,) * num_dim
    else:
        kernels = (k for k in kernel)

    dilations = None
    if isinstance(dilation, int):
        dilations = (dilation,) * num_dim
    else:
        dilations = (d for d in dilation)

    paddings = ()
    for d, k in zip(dilations, kernels):
        if k % 2 == 0 and d == 1:
            paddings += (d * (k) // 2,)
        else:
            paddings += (d * (k - 1) // 2,)
    logger.info(f"{paddings}")
    return paddings
