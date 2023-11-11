import sys

from einops import rearrange

sys.path.append(".")
import logging
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from src.datasets.base_dataset import find_class
from src.datautils.imbalanced_sampler import ImbalancedDatasetSampler
from src.datautils.stratified_sampler import StratifiedSampler
from torch.utils.data.dataloader import DataLoader
from src.context import get_project_root
from src.utils.slide_window import sliding_window
from src.datasets.NIOM.NIOM_spliter import NIOM_Spliter
from src.utils.normalization import get_norm_cls
from ..base_dataset import DictDataset, tensor_dataset

logger = logging.getLogger(__name__)
temp_folder = os.path.join(get_project_root(), ".temp")


def load_data(root, study_case):
    scenarios = {
        "case1": ["Home_A", "spring"],
        "case2": ["Home_A", "summer"],
        "case3": ["Home_B", "summer"],
    }
    household, season = scenarios[study_case]
    household_data = {}
    for entry, delim in zip(["energytrace", "person_1", "person_2"], ["\t", " ", " "]):
        fn = f"{household}_{season}_{entry}.txt"
        path = os.path.join(root, fn)
        household_data[entry] = pd.read_csv(path, delimiter=delim, header=None)
    return household_data


def preprocess_household_data(household_data):
    new_household_data = {}
    for p in ["person_1", "person_2"]:
        df_p = household_data[p].copy()
        df_p.columns = ["index", "date", "time", "occ"]
        # df_p = df_p.drop(['date', 'time'], axis=1)
        df_p = df_p.set_index("index")
        new_household_data[p] = df_p

    df_1 = new_household_data["person_1"]
    df_2 = new_household_data["person_2"]
    df = df_1.copy()
    # logger.info(df)
    df["occ"] = df_1["occ"] | df_2["occ"]
    new_household_data["occupancy"] = df
    df_e = household_data["energytrace"].copy()
    df_e.columns = ["index", "timestamp", "energy"]
    df_e = df_e.set_index("index")
    new_household_data["energytrace"] = df_e

    df_e = new_household_data["energytrace"]
    df_label = new_household_data["occupancy"]
    df = df_label.copy()
    df["energy"] = df_e["energy"]
    # fill na using backfill method
    df = df.fillna(method="backfill")
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_timedelta(df["time"] + ":00")
    df = df.set_index("time")
    df_by_date = [
        (pd.Timestamp(index), x.drop(["date"], axis=1)) for index, x in df.groupby("date")
    ]

    return df_by_date


def get_sampling_rate(df_data) -> pd.Timedelta:
    freq = df_data.index.to_series().diff().min()
    return freq


def get_aggregation_func(data, agg_freq, key, ori_freq):
    start, end = data.index[0], data.index[-1]
    time_range = pd.timedelta_range(start=start, end=end, freq=ori_freq).to_series(name=key)
    grouper = pd.Grouper(level=0, freq=agg_freq)
    return time_range, grouper


def aggregate(df_target, grouper, feature, time_range, date):
    agg_grp = df_target.groupby(grouper)
    df_fea = None
    if feature in ["mean", "std", "max", "min"]:
        df_fea = getattr(agg_grp, feature)()
    elif feature == "sad":

        def sad(x):
            def sad_c(x_c):
                a = np.arange(0, len(x_c))
                p = np.array(np.meshgrid(a, a)).T.reshape(-1, 2)
                y = np.abs(x_c.to_numpy()[p].T[0] - x_c.to_numpy()[p].T[1]).sum()
                return y

            return x.apply(sad_c)

        df_fea = agg_grp.apply(sad)
    elif feature == "autolag1":

        def autocor(x):
            def autocor_c(x_c):
                cor_lag_1 = np.correlate(x_c, x_c, mode="full")[1]
                return cor_lag_1

            return x.apply(autocor_c)

        df_fea = agg_grp.apply(autocor)
    elif feature == "range":
        df_fea = agg_grp.max() - agg_grp.min()
    elif feature == "raw":
        df_fea = agg_grp.mean()
    elif feature == "time":
        df_fea = pd.DataFrame(agg_grp.max().index.to_series())
        df_fea = df_fea.groupby(grouper).max()
        df_fea = df_fea / np.timedelta64(1, "s")
    elif feature == "week":
        weekday = date.weekday() / 2
        df_fea = pd.DataFrame(weekday, index=time_range, columns=["weekday"])
        df_fea = df_fea.groupby(grouper).max()
    elif feature == "hol":
        import holidays

        swiss_holidays = holidays.Switzerland()
        hol = int(date in swiss_holidays)
        df_fea = pd.DataFrame(hol, index=time_range, columns=["holiday"])
        df_fea = df_fea.groupby(grouper).max()
    elif feature == "fix":
        df_fea = None
    elif feature == "prob":
        df_fea = None
    elif feature == "cumsum":
        df_fea = None
    elif feature == "occ":
        df_fea = agg_grp.mean()
    return df_fea


def data_preprocess(date, ori_data, features, agg_time, slide_param) -> np.ndarray:
    freq = get_sampling_rate(ori_data)
    time_range, grouper = get_aggregation_func(ori_data, agg_time, "time", freq)
    features_list = []
    for fea in features:
        fea_data = ori_data.copy()
        df_fea = aggregate(fea_data, grouper, fea, time_range, date)
        if df_fea is not None:
            features_list.append(df_fea)
    np_features = pd.concat(features_list, axis=1).to_numpy()
    np_features = sliding_window(np_features, slide_param[0], slide_param[1])
    return np_features


def day_wrapper(data_by_date, key, features, agg_time, slide_param):
    all_data = []
    for date, ori_data in data_by_date:
        target_data = ori_data[[key]]
        data = data_preprocess(date, target_data, features, agg_time, slide_param)
        all_data.append(data)
    all_data = np.vstack(all_data)
    return all_data


# @buffer_value('joblib',temp_folder,disable=True)
def prepare_data(household_data, fea_type, agg_sec, slide_param, addtional_features):
    df_by_date = preprocess_household_data(household_data)
    if fea_type == "all":
        features = ["mean", "std", "max", "min", "sad", "autolag1", "range", *addtional_features]
    elif fea_type == "raw":
        features = ["mean", *addtional_features]

    agg_features = day_wrapper(
        df_by_date,
        "energy",
        features,
        agg_time=pd.Timedelta(agg_sec, "s"),
        slide_param=slide_param,
    )
    agg_occ = day_wrapper(
        df_by_date, "occ", ["occ"], agg_time=pd.Timedelta(agg_sec, "s"), slide_param=slide_param
    )
    # logger.debug(f'[occ] {agg_occ}')
    agg_features = rearrange(agg_features, "n t v -> n 1 v t")
    agg_occ = np.mean(agg_occ, axis=1).reshape(-1)
    # logger.debug(f'[occ] {agg_occ}')
    return agg_features, agg_occ


def get_norm_func(norm_type, fea_type):
    if fea_type == "cwt":
        mask_axis = (1,)
    else:
        mask_axis = (2,)
    return get_norm_cls(norm_type)(mask_axis)


def threshold_occ(y: np.ndarray, th):
    vfunc = np.vectorize(lambda x: float(x >= th))
    y_n = vfunc(y)
    return y_n


class NIOMDataset(pl.LightningDataModule):
    def __init__(
        self,
        study_case,
        norm_type,
        splits,
        fea_type,
        data_aug,
        agg_sec,
        batch_size,
        seed,
        imb_sam,
        slide_w,
        slide_s,
        t_en,
        w_en,
        hol_en,
        determ,
        num_workers,
    ):
        super().__init__()
        self.root = "./datasets/NIOM_occ/selected"
        self.study_case = study_case
        self.norm_type = norm_type
        self.splits = splits
        self.data_aug = data_aug
        self.agg_sec = agg_sec
        self.batch_size = batch_size
        self.seed = seed
        self.imbalance_sampler = imb_sam
        self.fea_type = fea_type
        self.slide_param = [slide_w, slide_s]
        self.addtional_features = []
        self.threshold = 0.5
        if t_en:
            self.addtional_features.append("time")
        if w_en:
            self.addtional_features.append("week")
        if hol_en:
            self.addtional_features.append("hol")
        self.determ = determ
        self.num_workers = num_workers
        self.fea_dict = {
            "as": agg_sec,
            "sw": slide_w,
            "ss": slide_s,
            "te": t_en,
            "we": w_en,
            "hol": hol_en,
        }

    @property
    def dataname(self):
        return self.datafea(self.fea_type)

    def datafea(self, fea_type):
        strf_fea = "_".join([f"{k}={v}" for k, v in self.fea_dict.items() if v != 0])
        return f"NIOM_{self.study_case}_{fea_type}_{strf_fea}"

    def set_repeation(self, repeat, nrepeat):
        self.repeat = repeat
        self.nrepeat = nrepeat

    def prepare_data(self) -> None:
        household_data = load_data(self.root, self.study_case)
        self.X, self.occ = prepare_data(
            household_data, self.fea_type, self.agg_sec, self.slide_param, self.addtional_features
        )
        self.y = threshold_occ(self.occ, self.threshold)
        self.nc = self.X.shape[1]
        self.variable_len = self.X.shape[2]
        self.max_len = self.X.shape[2]
        self.input_shape = [self.X.shape[1], self.X.shape[2], self.X.shape[3]]
        logger.debug(f"[Data shape] X shape {self.X.shape} y shape {self.y.shape}")

        self.labels, self.nclass, self.class_to_index = find_class(self.y)

        self.on_prepare_split()

    def on_prepare_split(self):
        self.spliter = NIOM_Spliter(
            self.dataname, self.labels, self.splits, self.nrepeat, deterministic=self.determ
        )
        train_index, val_index, test_index = self.spliter.get_split_repeat(self.repeat)
        self.train_x = self.X[train_index]
        self.train_y = self.y[train_index]
        self.val_x = self.X[val_index]
        self.val_y = self.y[val_index]
        self.test_x = self.X[test_index]
        self.test_y = self.y[test_index]

        # calcualte class weight
        count = np.bincount(self.labels[train_index])
        self.train_class_weight = (np.sum(count) - count) / np.sum(count)
        logger.info(f"Class weight {count} {self.train_class_weight}")

    def on_setup_normlize(self, data_x, data_y, stage=None):
        logger.debug(f"[norm_type] stage {stage} {self.norm_type}")
        if stage in (None, "fit"):
            (train_x, val_x), (train_y, val_y) = data_x, data_y
            if self.norm_type:
                norm_func = get_norm_func(self.norm_type, self.fea_type)
                train_x = norm_func.fit_transform(train_x)
                val_x = norm_func.transform(val_x)
                self.norm_func = norm_func

            return (train_x, val_x), (train_y, val_y)

        if stage in (None, "test", "predict"):
            (test_x), (test_y) = data_x, data_y
            if self.norm_type:
                test_x = self.norm_func.transform(test_x)

            return (test_x), (test_y)

    def on_data_augmentation(self, train_x, train_y):
        if self.determ:
            seed = self.seed
        else:
            seed = None
        if self.data_aug == "SMOTE":
            from src.datautils.oversample import DataAug_SMOTE

            smote = DataAug_SMOTE(random_state=seed)
            train_x, train_y = smote.resample(train_x, train_y)
            return train_x, train_y
        elif self.data_aug == "RANDOM":
            from src.datautils.oversample import DataAug_RANDOM

            rand_aug = DataAug_RANDOM(random_state=seed)
            train_x, train_y = rand_aug.resample(train_x, train_y)
            return train_x, train_y
        return train_x, train_y

    def setup(self, stage=None) -> None:
        if stage in (None, "fit"):
            (trn_x, val_x), (trn_y, val_y) = self.on_setup_normlize(
                (self.train_x, self.val_x), (self.train_y, self.val_y), stage
            )
            trn_x, trn_y = self.on_data_augmentation(trn_x, trn_y)
            # self.train_set = tensor_dataset(trn_x, trn_y)
            # self.val_set = tensor_dataset(val_x, val_y)
            self.train_set = DictDataset(trn_x, trn_y)
            self.val_set = DictDataset(val_x, val_y)

        if stage in (None, "test", "predict"):
            (test_x), (test_y) = self.on_setup_normlize((self.test_x), (self.test_y), stage)
            # self.test_set = tensor_dataset(test_x, test_y)
            self.test_set = DictDataset(test_x, test_y)

    def _to_dataloader(self, dataset, shuffle, batch_size, num_workers, drop_last, sampler=None):
        if sampler:
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
        )

    def train_dataloader(self):
        sampler = None
        if self.imbalance_sampler:
            sampler = ImbalancedDatasetSampler(self.train_set)
        return self._to_dataloader(
            self.train_set,
            True,
            self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        return self._to_dataloader(
            self.val_set, False, self.batch_size, num_workers=self.num_workers, drop_last=False
        )

    def test_dataloader(self):
        return self._to_dataloader(
            self.test_set, False, self.batch_size, num_workers=self.num_workers, drop_last=False
        )

    def predict_dataloader(self):
        return self._to_dataloader(
            self.test_set, False, self.batch_size, num_workers=self.num_workers, drop_last=False
        )


def test():
    from src.logger.get_configured_logger import get_console_logger

    my_logger = get_console_logger()
    keys_dict = dict(
        study_case="case1",
        norm_type="minmax",
        splits="3:1:1",
        fea_type="raw",
        data_aug="SMOTE",
        agg_sec=60,
        batch_size=32,
        random_state=42,
        imb_sam=1,
        slide_w=60,
        slide_s=60,
        t_en=1,
        w_en=1,
        hol_en=0,
        deterministic=1,
    )
    niom = NIOMDataset(**keys_dict)
    niom.set_repeation(1, 10)
    niom.prepare_data()
    niom.setup("fit")
    for x, y in niom.train_dataloader():
        logger.info(f"{x.shape} {y.shape}")
        exit()


if __name__ == "__main__":
    test()
