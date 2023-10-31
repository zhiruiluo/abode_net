import numpy as np
import time
import random
import sys
sys.path.append('.')
from src.datautils.data_spliter import IndexSpliter


class ECO_Spliter(IndexSpliter):
    def __init__(self, key_name, labels, splits, nrepeat, deterministic=True) -> None:
        super().__init__(key_name, labels, splits, nrepeat, deterministic)

    def _splits_(self, labels):
        from sklearn.model_selection import train_test_split
        if self.splits == '3:1:1':
            # seed = 42
            random.seed(time.time())
            seed = random.randint(0,100000)
            train_index, test_index = train_test_split(np.arange(len(labels)),train_size=0.8,stratify=labels, shuffle=True,random_state=seed)
            train_labels = labels[train_index]
            train_index, val_index = train_test_split(train_index,train_size=0.75,stratify=train_labels, shuffle=True,random_state=seed)
            
        return train_index, val_index, test_index


# TODO 3:1:1 spliter by date

if __name__ == '__main__':
    label = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    nrepeat = 10
    spliter = ECO_Spliter('test', label, '3:1:1', nrepeat, deterministic=False)
    for repeat in range(nrepeat):
        print(spliter.get_split_repeat(repeat))
