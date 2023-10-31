import pickle
import os
import pandas as pd
import marshal
import inspect
import joblib
from typing import Any
from time import time
import logging
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ['buffer_value', 'disk_buffer', 'timeit']

def pandas_to_file(df: pd.DataFrame, path: str):
    df.to_csv(path)

def pandas_from_file(path: str):
    return pd.read_csv(path)

def pickle_to_file(object: Any, path: str):
    pickle.dump(object,open(path,'wb'))

def pickle_from_file(path: str):
    return pickle.load(open(path,'rb'))

def joblib_to_file(object: Any, path: str, compress: int = 0):
    joblib.dump(object, path, compress=compress)

def joblib_from_file(path: str):
    return joblib.load(path)

def protocol_writer(protocol):
    if protocol == 'pandas':
        return pandas_to_file
    elif protocol == 'pickle':
        return pickle_to_file
    elif protocol == 'joblib':
        return joblib_to_file
    else:
        raise ValueError(f'buffer_value protocol_writer Error: get {protocol}!')

def protocol_reader(protocol):
    if protocol == 'pandas':
        return pandas_from_file
    elif protocol == 'pickle':
        return pickle_from_file
    elif protocol == 'joblib':
        return joblib_from_file
    else:
        raise ValueError(f'buffer_value protocol_reader Error: get {protocol}!')
    
def protocol_postfix(protocol):
    if protocol == 'pandas':
        return '.csv'
    elif protocol == 'pickle':
        return '.pickle'
    elif protocol == 'joblib':
        return '.pkl'
    else:
        raise ValueError(f'buffer_value protocol_postfix Error: get {protocol}!')

def buffer_value(protocol, folder, disable=False):
    '''decorator for buffering temporary values in files\n
    protocol: [ 'pandas' | 'pickle' | 'joblib' ]\n
    folder: user defined path
    '''
    def decorator(func):
        def BufferWrapper(fn, *args, **kwargs):
            logger.debug(f'[buffer_value] {folder} {fn}')
            if not os.path.isdir(folder):
                os.mkdir(folder)
            
            fpath = os.path.join(os.path.join(folder,fn+protocol_postfix(protocol)))
            
            def run_and_write():
                func_code = inspect.getsource(func)
                out = func(*args,**kwargs)
                if not disable:
                    writer = protocol_writer(protocol)
                    writer((func_code,out), fpath)
                    logger.debug(f'Writer object to {fpath}, @{protocol}, spent time {time()-t1:2.4f} sec')
                return out

            t1 = time()
            rerun_flag = False
            if not os.path.isfile(fpath) or disable:
                rerun_flag = True
            else:
                reader = protocol_reader(protocol)
                try:
                    prev_func_code, out = reader(fpath)
                    func_code = inspect.getsource(func)
                    if prev_func_code != func_code:
                        logger.debug(f'function_code change detected! {len(prev_func_code)} {len(func_code)} {prev_func_code != func_code}')
                        rerun_flag = True
                    else:
                        logger.debug(f'Read object from {fpath} in FUNC {func.__name__} {inspect.getfile(func)} #{inspect.currentframe().f_back.f_lineno}, @{protocol}, spent time {time()-t1:2.4f} sec')
                except Exception as e:
                    logger.debug(f'[buffer_value] {traceback.format_exc()}. Rerun!')
                    rerun_flag = True                
            if rerun_flag:
                out= run_and_write()
            return out
        return BufferWrapper
    return decorator


class disk_buffer(object):
    def __init__(self, func, protocol='pickle', folder='.temp', disable=False) -> None:
        # self.buffer_fn = buffer_fn
        self.protocol = protocol
        self.folder = Path(folder)
        self.disable = disable
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._buffer_key(*args, **kwargs)
        self.folder.mkdir(parents=True, exist_ok=True)
        fpath = self.folder.joinpath(key+protocol_postfix(self.protocol))
        
        def run_and_write():
            func_code = inspect.getsource(self._func)
            out = self._func(*args,**kwargs)
            if not self.disable:
                writer = protocol_writer(self.protocol)
                writer((func_code,out), fpath)
                logger.debug(f'Writer object to {fpath}, @{self.protocol}, spent time {time()-t1:2.4f} sec')
            return out
        
        t1 = time()
        rerun_flag = False
        if fpath.is_file() or self.disable:
            rerun_flag = True
        else:
            reader = protocol_reader(self.protocol)
            try:
                prev_func_code, out = reader(fpath)
                func_code = inspect.getsource(self.func)
                if prev_func_code != func_code:
                    logger.debug(f'function_code change detected! {len(prev_func_code)} {len(func_code)} {prev_func_code != func_code}')
                    rerun_flag = True
                else:
                    logger.debug(f'Read object from {fpath} in FUNC {self.func.__name__} {inspect.getfile(self.func)} #{inspect.currentframe().f_back.f_lineno}, @{self.protocol}, spent time {time()-t1:2.4f} sec')
            except Exception as e:
                logger.debug(f'[buffer_value] {traceback.format_exc()}. Rerun!')
                rerun_flag = True          
        if rerun_flag:
            out = run_and_write()
        return out
        
    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        pass

    def _buffer_key(self, *args, **kwargs):
        return pickle.dumps((args, sorted(kwargs.items())), pickle.HIGHEST_PROTOCOL)


def timeit(func):
    def timer_wrapper(*args, **kwargs):
        t1 = time()
        out = func(*args, **kwargs)
        logger.info(f'Function {func.__name__} args: [{args}, {kwargs}] spent time {time()-t1:2.4f} sec')
        return out
    return timer_wrapper