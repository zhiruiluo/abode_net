import torch 
import logging
import os

logger = logging.getLogger(__name__)

def get_num_gpus():
    num_gpus = torch.cuda.device_count()
    logger.info(f'[num_gpus] {num_gpus}')
    return num_gpus

def cpu_count():
    if os.environ.get('SLURM_CPUS_ON_NODE'):
        cpus_reserved = int(os.environ['SLURM_CPUS_ON_NODE'])
    else:
        cpus_reserved = 8
    return cpus_reserved
    # return multiprocessing.cpu_count()

def cuda_available():
    if get_num_gpus() == 0:
        return False
    return True