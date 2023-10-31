import sys

sys.path.append(".")
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch import tensor
from ..base_dataset import find_class
from .ECO_utils import (
    preprocess,
    threshold_occ,
    load_data,
    retrive_study_case,
    get_norm_func,
)
import os
from src.sampler.imbalanced_sampler import ImbalancedDatasetSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from src.context import get_project_root
from .ECO_spliter import ECO_Spliter

logger = logging.getLogger(__name__)


class ECODataset(pl.LightningDataModule):
    def __init__(
        self,
        study_case,
        fea_type,
        agg_sec,
        slide_w,
        slide_s,
        splits,
        norm_type,
        batch_size,
        seed,
        imb_sam,
        t_en,
        w_en,
        hol_en,
        fix_en,
        prob_en,
        plugs_en,
        plugs_cum,
        sm_cum,
        data_aug,
        determ,
        num_workers=8,
    ) -> None:
        super().__init__()
        self.feature_type = fea_type
        self.aggregation_sec = agg_sec
        self.slide_winsize = slide_w
        self.slide_stride = slide_s
        self.splits = splits
        self.norm_type = norm_type
        self.batch_size = batch_size
        self.seed = seed
        self.imbalance_sampler = imb_sam
        self.time_encoding = t_en
        self.week_encoding = w_en
        self.fix_encoding = fix_en
        self.prob_encoding = prob_en
        self.holiday_encoding = hol_en
        self.plugs_encoding = plugs_en
        self.plugs_cum = plugs_cum
        self.sm_cum = sm_cum
        self.data_aug = data_aug
        self.determ = determ
        self.num_workers = num_workers
        self._study_case = study_case
        self.root = os.path.join(get_project_root(), "datasets/eco/")

        self.fea_dict = {
            "as": self.aggregation_sec,
            "sw": self.slide_winsize,
            "ss": self.slide_stride,
            "te": self.time_encoding,
            "we": self.week_encoding,
            "hol": self.holiday_encoding,
            "fix": self.fix_encoding,
            "prob": self.prob_encoding,
            "plugs": self.plugs_encoding,
            "plugscum": self.plugs_cum,
            "sm_cum": self.sm_cum,
        }

    @property
    def dataname(self):
        return self.datafea(self.feature_type)

    def datafea(self, fea_type):
        strf_fea = "_".join([f"{k}={v}" for k, v in self.fea_dict.items() if v != 0])
        return f"ECO_{self._study_case}_{fea_type}_{strf_fea}"

    def set_repeation(self, repeat, nrepeat):
        self.repeat = repeat
        self.nrepeat = nrepeat

    def prepare_data(self) -> None:
        logger.info(f"[Prepare Dataset] {self.dataname}")
        self.study_case = retrive_study_case(self._study_case)
        self.household_data = load_data(
            f"ECO_{self._study_case}",
            self.root,
            self.study_case["household"],
            self.study_case["period"],
        )

        household = self.household_data[self.study_case["household"]]
        th = self.study_case["threshold"]

        if self.feature_type == "cwtraw":
            fea_type = "cwt"
            self.X, self.occ = preprocess(
                f"{self.datafea(fea_type)}_preprocessed",
                household,
                th,
                fea_type,
                self.aggregation_sec,
                self.slide_winsize,
                self.time_encoding,
                self.week_encoding,
            )
            fea_type1 = "raw"
            self.X1, _ = preprocess(
                f"{self.datafea(fea_type1)}_preprocessed",
                household,
                th,
                fea_type1,
                self.aggregation_sec,
                self.slide_winsize,
                self.time_encoding,
                self.week_encoding,
            )
            self.y = threshold_occ(self.occ, self.study_case["threshold"])
        else:
            preprocess_cache_name = f"{self.dataname}_preprocessed"
            self.X, self.occ = preprocess(
                preprocess_cache_name,
                household,
                th,
                self.feature_type,
                self.aggregation_sec,
                self.slide_winsize,
                self.slide_stride,
                self.time_encoding,
                self.week_encoding,
                self.holiday_encoding,
                self.fix_encoding,
                self.prob_encoding,
                self.plugs_encoding,
                self.plugs_cum,
                self.sm_cum,
            )
            self.y = threshold_occ(self.occ, self.study_case["threshold"])

        self.nc = self.X.shape[1]
        self.variable_len = self.X.shape[2]
        self.max_len = self.X.shape[3]
        self.input_shape = [self.X.shape[1], self.X.shape[2], self.X.shape[3]]
        logger.info(f"[Data shape] X shape {self.X.shape} y shape {self.y.shape}")

        # find class
        self.labels, self.nclass, self.class_to_index = find_class(self.y)

        # Splite dataset
        self.on_prepare_split()

    def on_prepare_split(self):
        self.spliter = ECO_Spliter(
            self.dataname,
            self.labels,
            self.splits,
            self.nrepeat,
            deterministic=self.determ,
        )
        train_index, val_index, test_index = self.spliter.get_split_repeat(self.repeat)
        self.train_x = self.X[train_index]
        self.train_y = self.y[train_index]
        self.val_x = self.X[val_index]
        self.val_y = self.y[val_index]
        self.test_x = self.X[test_index]
        self.test_y = self.y[test_index]
        if self.feature_type == "cwtraw":
            self.train_x1 = self.X1[train_index]
            self.val_x1 = self.X1[val_index]
            self.test_x1 = self.X1[test_index]

        # calcualte class weight
        count = np.bincount(self.labels[train_index])
        self.train_class_weight = (np.sum(count) - count) / np.sum(count)
        logger.info(f"Class weight {count} {self.train_class_weight}")

    def on_setup_normlize(self, stage=None):
        logger.info(f"[norm_type] stage {stage} {self.norm_type}")
        if stage in (None, "fit"):
            if not self.norm_type:
                if self.feature_type == "cwtraw":
                    return (self.train_x, self.train_x1, self.train_y), (
                        self.val_x,
                        self.val_x1,
                        self.val_y,
                    )
                else:
                    return (self.train_x, self.train_y), (self.val_x, self.val_y)

            if self.feature_type == "cwtraw":
                fea_type = "cwt"
                self.norm_func = get_norm_func(self.norm_type, "cwt")
                train_x = self.norm_func.fit_transform(self.train_x)
                val_x = self.norm_func.transform(self.val_x)

                self.norm_func1 = get_norm_func(self.norm_type, "raw")
                train_x1 = self.norm_func1.fit_transform(self.train_x1)
                val_x1 = self.norm_func1.transform(self.val_x1)
                return (train_x, train_x1, self.train_y), (val_x, val_x1, self.val_y)
            else:
                fea_type = self.feature_type
                self.norm_func = get_norm_func(self.norm_type, fea_type)
                train_x = self.norm_func.fit_transform(self.train_x)
                val_x = self.norm_func.transform(self.val_x)
                return (train_x, self.train_y), (val_x, self.val_y)

        if stage in (None, "test", "predict"):
            if not self.norm_type:
                if self.feature_type == "cwtraw":
                    return (self.test_x, self.text_x1, self.test_y)
                else:
                    return (self.test_x, self.test_y)
            if self.feature_type == "cwtraw":
                test_x = self.norm_func.transform(self.test_x)
                test_x1 = self.norm_func1.transform(self.test_x1)
                return (test_x, test_x1, self.test_y)
            else:
                test_x = self.norm_func.transform(self.test_x)
                return (test_x, self.test_y)

    def on_data_augmentation(self, train_x, train_y):
        if self.determ:
            seed = self.seed
        else:
            seed = None
        if self.data_aug == "SMOTE":
            from src.dataset.data_augmentation import DataAug_SMOTE

            smote = DataAug_SMOTE(random_state=seed)
            train_x, train_y = smote.resample(train_x, train_y)
            return train_x, train_y
        elif self.data_aug == "RANDOM":
            from src.dataset.data_augmentation import DataAug_RANDOM

            rand_aug = DataAug_RANDOM(random_state=seed)
            train_x, train_y = rand_aug.resample(train_x, train_y)
            return train_x, train_y
        return train_x, train_y

    def setup(self, stage=None) -> None:
        t_f, t_l = torch.float32, torch.long
        if stage in (None, "fit"):
            if self.feature_type == "cwtraw":
                (trn_x, trn_x1, trn_y), (val_x, val_x1, val_y) = self.on_setup_normlize(
                    stage
                )
                self.train_set = TensorDataset(
                    tensor(trn_x, dtype=t_f),
                    tensor(trn_x1, dtype=t_f),
                    tensor(trn_y, dtype=t_l),
                )
                self.val_set = TensorDataset(
                    tensor(val_x, dtype=t_f),
                    tensor(val_x1, dtype=t_f),
                    tensor(val_y, dtype=t_l),
                )
            else:
                (trn_x, trn_y), (val_x, val_y) = self.on_setup_normlize(stage)
                trn_x, trn_y = self.on_data_augmentation(trn_x, trn_y)
                self.train_set = TensorDataset(
                    tensor(trn_x, dtype=t_f), tensor(trn_y, dtype=t_l)
                )
                self.val_set = TensorDataset(
                    tensor(val_x, dtype=t_f), tensor(val_y, dtype=t_l)
                )

        if stage in (None, "test", "predict"):
            if self.feature_type == "cwtraw":
                (test_x, test_x1, test_y) = self.on_setup_normlize(stage)
                self.test_set = TensorDataset(
                    tensor(test_x, dtype=t_f),
                    tensor(test_x1, dtype=t_f),
                    tensor(test_y, dtype=t_l),
                )
            else:
                (test_x, test_y) = self.on_setup_normlize(stage)
                self.test_set = TensorDataset(
                    tensor(test_x, dtype=t_f), tensor(test_y, dtype=t_l)
                )

    def _to_dataloader(self, dataset, shuffle, batch_size, drop_last, sampler=None):
        if sampler:
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            # collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader

    def train_dataloader(self):
        if self.imbalance_sampler:
            sampler = ImbalancedDatasetSampler(self.train_set)
        else:
            sampler = None
        return self._to_dataloader(
            self.train_set, True, self.batch_size, drop_last=False, sampler=sampler
        )

    def val_dataloader(self):
        return self._to_dataloader(
            self.val_set, False, self.batch_size, drop_last=False
        )

    def test_dataloader(self):
        return self._to_dataloader(
            self.test_set, False, self.batch_size, drop_last=False
        )

    def predict_dataloader(self):
        return self._to_dataloader(
            self.test_set, False, self.batch_size, drop_last=False
        )


def test1():
    from src.logger.get_configured_logger import get_console_logger

    mylogger = get_console_logger(False)

    for t_en in [1]:
        fea_type = "raw"
        agg_sec = 1
        slide_w = 15 * 60
        slide_s = 60
        for case in range(1, 2):
            c = f"case{case}"
            eco_data = ECODataset(
                c,
                fea_type,
                agg_sec,
                slide_w,
                slide_s,
                "3:1:1",
                "minmax",
                32,
                42,
                0,
                t_en,
                t_en,
                True,
            )
            eco_data.prepare_data()
            eco_data.setup()
            # eco_data.visualize()
            # for i, batch in enumerate(eco_data.val_dataloader()):
            #     x,y = batch
            #     eco_data.visualize_batch(x, y, y, i)


# if __name__ == '__main__':


#     test1()
