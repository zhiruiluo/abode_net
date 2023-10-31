import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

logger = logging.getLogger(__name__)

def encode_onehot(labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot, classes_dict

def get_datapath(root, dataname):
    datapath = os.path.join(root, dataname)
    return datapath

def find_class(labels: Union[np.ndarray, List]):
    classes = sorted(np.unique(labels))
    class_to_index = {classname: i for i,
                            classname in enumerate(classes)}
    logger.info(f'class_to_index {class_to_index}')
    nclass = len(classes)
    index = np.vectorize(class_to_index.__getitem__)(labels)
    if len(index.shape) == 2:
        index = index.reshape(-1)
    logger.info(f'Label counts: {list(enumerate(np.bincount(index)))}')
    return index, nclass, class_to_index

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def key_name(self):
        return self.dataname

    def standard_norm(self, train, val=None, test=None):
        ss_train, ss_val, ss_test = None, None, None
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss_train = ss.fit_transform(train)
        if val is not None:
            ss_val = ss.transform(val)
        if test is not None:
            ss_test = ss.transform(test)
        
        return ss_train, ss_val, ss_test

    def _find_class_(self, labels: Union[np.ndarray, List], one_hot: bool) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """convert class to indexes"""
        classes = sorted(np.unique(labels))
        self.class_to_index = {classname: i for i,
                                classname in enumerate(classes)}
        logger.info(f'class_to_index { self.class_to_index}')
        self.class_names = classes
        self.nclass = len(classes)
        self.indexes = [i for i in range(len(self.class_names))]
        index = np.vectorize(self.class_to_index.__getitem__)(labels)
        if len(index.shape) == 2:
            index = index.reshape(-1)
        logger.info(f'Label counts: {list(enumerate(np.bincount(index)))}')
        if one_hot:
            labels_onehot, classes_dict = encode_onehot(labels)
            self.classes_dict = classes_dict
            logger.info(f'classes_dict {self.classes_dict}')
            return index, labels_onehot
        
        return index, None


class BaseSplitDataset(BaseDataset):
    def __init__(self, dataname):
        super().__init__()
        self.root = 'datasets'
        self.dataname = dataname

    def maxlen(self):
        x_train = self.train_data_ndarray['X_train']
        return x_train.shape[2]

    def _load(self, phase):
        datapath = get_datapath(self.root, self.dataname)
        data_ndarray = {}
        for ds in ['X','y']:
            data_ndarray[f'{ds}_{phase}'] = np.load(os.path.join(datapath, f'{ds}_{phase}.npy'))
        return data_ndarray

    def get_split(self):
        self.train_data_ndarray = self._load('train')
        self.test_data_ndarray = self._load('test')

        x_train = torch.tensor(self.train_data_ndarray['X_train'], dtype=torch.float32)
        y_train = torch.tensor(self.train_data_ndarray['y_train'])
        x_test = torch.tensor(self.test_data_ndarray['X_test'], dtype=torch.float32)
        y_test = torch.tensor(self.test_data_ndarray['y_test'])

        self.variable_len = x_train.shape[1]

        # ss_train, _, ss_test = self.standard_norm(x_train, x_test)

        label_train, _ = self._find_class_(y_train, False)
        label_test, _ = self._find_class_(y_test, False)

        label_train = torch.tensor(label_train, dtype=torch.long)
        label_test = torch.tensor(label_test, dtype=torch.long)
        logger.info(f'x_train {x_train.shape} label_train {label_train.shape} x_test {x_test.shape} label_test {label_test.shape}')
        return TensorDataset(x_train, label_train), TensorDataset(x_test, label_test), TensorDataset(x_test, label_test)
