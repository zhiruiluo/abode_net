import os
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Subset(Dataset):
    def __init__(self, dataset, indices, transform=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, index):
        x, label =self.dataset[self.indices[index]]
        if self.transform:
            x = self.transform(x)
        return x, label
    
    def __len__(self):
        return len(self.indices)


class Spliter():
    def __init__(self, dataset, normal, deterministic=True) -> None:
        self.dataset = dataset
        self.normal = normal
        self.deterministic = deterministic
        self._index_save_path = './spliting/'

    def _splits_(self, y, splits):
        train_index, val_index, test_index = None,None,None
        yield train_index, val_index, test_index

    def get_split(self):
        split_index = self._setup_split_index()
        train, val, test = self._index_to_subset(split_index, self.dataset)
        logger.info(f'[spliter summary] (splits {self.splits}) train {len(train)} val {len(val)} test {len(test)}')
        return train, val, test

    def _setup_split_index(self):
        if not os.path.isdir(self._index_save_path):
            os.mkdir(self._index_save_path)

        path = os.path.join(self._index_save_path, f'spliting_{self.dataset.key_name}_{self.splits}.csv')
        if not os.path.isfile(path) or not self.deterministic:
            train_idx, val_idx, test_idx = self._splits_()
            train_idx = [(i, 0) for i in train_idx]
            test_idx = [(i, 2) for i in test_idx]

            df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
            df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
            if val_idx is not None:
                val_idx = [(i,1) for i in val_idx]
                df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
                df_tuple = (df_train, df_val, df_test)
            else:
                df_tuple = (df_train, df_test)
            
            df_split_index = pd.concat(df_tuple, axis=0, ignore_index=True)
            if self.deterministic:
                df_split_index.to_csv(path, index=False)
            return df_split_index
        else:
            df_split_index = pd.read_csv(path)
            return df_split_index
        
    def _index_to_subset(self, split_index, dataset):
        train_index = split_index[split_index['train_type']==0]['index'].to_numpy()
        val_index = split_index[split_index['train_type']==1]['index'].to_numpy()
        test_index = split_index[split_index['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        train = Subset(dataset, train_index)
        val = Subset(dataset, val_index)
        test = Subset(dataset, test_index)
        train_count = self._count_by_class('train',train)
        val_count = self._count_by_class('val',val)
        test_count = self._count_by_class('test',test)

        self.train_class_weight = np.array(train_count)/np.sum(train_count)
        self.val_count_weight = np.array(val_count)/np.sum(val_count)
        self.test_count_weight = np.array(test_count)/np.sum(test_count)

        return train, val, test

    def _count_by_class(self, phase, dataset):
        count = defaultdict(int)
        for data in dataset:
            if isinstance(data, dict):
                i_cls = data['label']
            else:
                i_cls = data[-1]
            
            if isinstance(i_cls, int):
                count[i_cls] += 1
            else:
                count[i_cls.item()] += 1
            
        logger.info('[{} class] {}'.format(phase, [ (k,v) for k,v in \
            sorted(count.items(), key=lambda item: item[0])]))

        return list(count.values())

class IndexSpliter():
    def __init__(self, key_name, labels, splits: str, nrepeat, deterministic=True) -> None:
        self.key_name = key_name
        self.labels = labels
        self.splits = splits
        self.nrepeat = nrepeat
        self.deterministic = deterministic
        self._index_save_path = './spliting/'

    def _splits_(self, labels):
        train_index, val_index, test_index = None,None,None
        yield train_index, val_index, test_index

    # def prepare(self):
    #     self._setup_split_index()

    def get_split(self):
        train_index, val_index, test_index = self._setup_split_index()
        logger.debug(f'[spliter summary] (splits {self.splits}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def get_split_repeat(self, repeat):
        self.repeat = repeat
        train_index, val_index, test_index = self._setup_split_index()
        logger.debug(f'[spliter summary] (splits {self.splits}) train {len(train_index)} val {len(val_index)} test {len(test_index)}')
        return train_index, val_index, test_index

    def split_and_save_index(self, path):
        train_idx, val_idx, test_idx = self._splits_(self.labels)
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_split_index = pd.concat(df_tuple, axis=0, ignore_index=True)
        if self.deterministic:
            df_split_index.to_csv(path, index=False)
            logger.debug(f'[spliter] index created and saved!')
        else:
            logger.debug(f'[spliter] index created without saving to file!')
        
        return df_split_index

    def _setup_split_index(self):
        if not os.path.isdir(self._index_save_path):
            os.mkdir(self._index_save_path)

        folder = f'splt_{self.nrepeat}_{self.key_name}_{self.splits}/'
        path = Path(self._index_save_path).joinpath(folder)
        path.mkdir(parents=True,exist_ok=True)
        if self.nrepeat <= 1:
            path = path.joinpath(f'index.csv')
        else:
            path = path.joinpath(f'index_{self.repeat}.csv')
        if not os.path.isfile(path) or not self.deterministic:
            df_split_index = self.split_and_save_index(path)
        else:
            df_split_index = pd.read_csv(path)
            if len(df_split_index) != len(self.labels):
                logger.debug(f'[spliter] df_split_index and labels length not match {len(df_split_index)} != {len(self.labels)}')
                df_split_index = self.split_and_save_index(path)
            else:
                logger.debug(f'[spliter] Read index from file, length of index {len(df_split_index)}')

        return self._df_to_index(df_split_index)
        
    def _df_to_index(self, df_split_index):
        train_index = df_split_index[df_split_index['train_type']==0]['index'].to_numpy()
        val_index = df_split_index[df_split_index['train_type']==1]['index'].to_numpy()
        test_index = df_split_index[df_split_index['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        return train_index, val_index, test_index