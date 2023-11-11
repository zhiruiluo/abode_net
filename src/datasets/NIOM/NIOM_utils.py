import zipfile
from src.utils.decorator import buffer_value
import time
import pandas as pd
import numpy as np
import re
import os
import io
import logging
from src.utils.slide_window import sliding_window, span_to_array
from src.utils.normalization import get_norm_cls
from tqdm import tqdm
from einops import rearrange
from scipy import signal
import holidays
from datetime import timedelta
from src.context import get_project_root

disable = False

logger = logging.getLogger(__name__)
temp_folder = os.path.join(get_project_root(), ".temp")


def extract_zip(input_zip):
    input_zip = zipfile.ZipFile(input_zip)
    return {
        name: input_zip.read(name)
        for name in input_zip.namelist()
        if (".csv" in name) & ("__MACOSX" not in name)
    }


def read_occupancy_csv(buffer):
    rows = []
    indexes = []
    for i, line in enumerate(buffer):
        line = line.decode("utf-8")
        if i == 0:
            continue
        else:
            l = line.strip("\n")
            cols = re.split(",", l)
            indexes.append(cols[0])
            row = [int(v) for v in cols[1:]]
            rows.append(row)

    return pd.DataFrame(np.array(rows), index=pd.DatetimeIndex(indexes))


@buffer_value("joblib", temp_folder)
def load_data(root, household, period):
    zip_files = [
        f"{household}_sm_csv.zip",
        f"{household}_occupancy_csv.zip",
        f"{household}_plugs_csv.zip",
    ]
    house_hold_df = {}
    for f in zip_files:
        if household not in house_hold_df.keys():
            house_hold_df[household] = {"occupancy": {}, "sm": {}}
        unzip_files = extract_zip(os.path.join(root, f))

        t1 = time.time()
        for k, v in unzip_files.items():
            s = io.BytesIO(v)
            if "occupancy_csv" in f:
                date = k.split("/")[-1].rstrip(".csv")
                if period not in date:
                    continue
                df = read_occupancy_csv(s)
                house_hold_df[household]["occupancy"] = df
            elif "sm_csv" in f:
                name = k.split("/")[-1].rstrip(".csv")
                df = pd.read_csv(s, header=None, index_col=None, dtype=float)
                house_hold_df[household]["sm"][name] = df
            elif "plugs_csv" in f:
                hshld, appliance, date_csv = k.split("/")
                date = date_csv.rstrip(".csv")
                df = pd.read_csv(s, header=None, index_col=None, dtype=float)
                appliance = f"plugs_{appliance}"
                if appliance not in house_hold_df[household]:
                    house_hold_df[household][appliance] = {}
                house_hold_df[household][appliance][date] = df
            else:
                continue
        t2 = time.time()
        logger.info(f"{t2-t1} s")
    return house_hold_df


def retrive_study_case(case):
    cases = {
        "case1": {
            "household": "01",
            "period": "summer",
            "threshold": 0.5,
        },
        "case2": {
            "household": "01",
            "period": "winter",
            "threshold": 0.5,
        },
        "case3": {
            "household": "02",
            "period": "winter",
            "threshold": 0.5,
        },
        "case4": {
            "household": "03",
            "period": "summer",
            "threshold": 0.5,
        },
    }

    if case in cases:
        return cases[case]


def group_feature(df, tile, operation):
    df["group"] = tile
    encoding = getattr(df.groupby(["group"]), operation)()
    return encoding


def aggregate_sliding_window_by_date(df, aggregation_seconds, slide_winsize, slide_stride):
    time_range = pd.date_range(
        "00:00:00", periods=24 * 60 * 60 / aggregation_seconds, freq=f"{aggregation_seconds}s"
    ).strftime("%H:%M:%S")
    if df is None:
        df = pd.DataFrame([0] * 24 * 60 * 60)
    tile = np.tile(time_range, (aggregation_seconds, 1)).T.reshape(-1)
    tile = pd.Series(tile, name="group", index=df.index)
    np_agg = group_feature(df, tile, "mean").to_numpy()

    # ((n 60) v) -> (n 60 v)
    np_agg_slided = sliding_window(np_agg, slide_winsize, slide_stride)
    return np_agg_slided


def aggregate_appliance(appl_comm, aggregation_seconds, slide_winsize, slide_stride, cumsum):
    all_dates = []
    for date, df_appl_by_date in tqdm(appl_comm):
        # (n sw v)
        features = []
        if df_appl_by_date is None:
            features.append(pd.DataFrame([0] * 24 * 60 * 60))
        else:
            features.append(df_appl_by_date)
        if cumsum:
            if df_appl_by_date is None:
                df_cum = pd.DataFrame([0] * 24 * 60 * 60)
            else:
                df_cum = df_appl_by_date.cumsum()
            features.append(df_cum)

        features = pd.concat(features, axis=1)
        all_dates.append(
            aggregate_sliding_window_by_date(
                features, aggregation_seconds, slide_winsize, slide_stride
            )
        )

    return np.vstack(all_dates)


def aggregate_occupancy(df_occ_comm, aggregation_seconds, slide_winsize, slide_stride, th):
    all_occ = []
    df_occ_comm = df_occ_comm.T
    for date in df_occ_comm:
        occ_by_date = df_occ_comm[[date]].copy()
        occ = aggregate_sliding_window_by_date(
            occ_by_date, aggregation_seconds, slide_winsize, slide_stride
        )
        occ = np.mean(occ, axis=1)
        all_occ.append(occ)
    all_oc = np.vstack(all_occ)
    return all_oc.reshape(-1)


def aggregate_sm_cwt(sm_comm, aggregation_seconds, sliding_window, f_time, f_week):
    all_features = []
    for date, sm_df_by_date in tqdm(sm_comm):
        temp_df = sm_df_by_date.loc[:, 0:3].copy()
        time_range = pd.date_range(
            "00:00:00", periods=24 * 60 * 60 / aggregation_seconds, freq=f"{aggregation_seconds}s"
        ).strftime("%H:%M:%S")
        tile = np.tile(time_range, (aggregation_seconds, 1)).T.reshape(-1)
        tile = pd.Series(tile, name="group", index=temp_df.index)
        temp_df["group"] = tile

        features = []

        four_features = rearrange(
            temp_df.groupby(["group"]).mean().to_numpy(), "(n w) v -> n v w", w=sliding_window
        )
        cwt_feature = []
        for n in four_features:
            samp_fea = []
            for fea in n:
                samp_fea.append(
                    signal.cwt(fea, signal.ricker, widths=np.arange(1, aggregation_seconds + 1))
                )
            samp_fea = np.stack(samp_fea)
            cwt_feature.append(samp_fea)
        cwt_feature = np.stack(cwt_feature)
        features.append(cwt_feature)

        if f_time:
            df_time = pd.DataFrame(np.arange(24 * 60 * 60), columns=["time"], index=temp_df.index)
            df_time["group"] = tile
            np_time = df_time.groupby(["group"]).max().to_numpy()
            time_feature = rearrange(np_time, "(n w) 1 -> n w", w=sliding_window)

            time_encoding = []
            for row in time_feature:
                time_encoding.append(np.vstack([row] * aggregation_seconds))
            time_encoding = np.stack(time_encoding)
            time_encoding = rearrange(time_encoding, "n w1 w2 -> n 1 w1 w2")
            features.append(time_encoding)

        if f_week:
            weekday = date.weekday()
            df_weekday = pd.DataFrame(
                [weekday] * 24 * 60 * 60, columns=["weekday"], index=temp_df.index
            )
            df_weekday["group"] = tile
            week_feature = rearrange(
                df_weekday.groupby(["group"]).max().to_numpy(), "(n w) 1 -> n w", w=sliding_window
            )

            week_encoding = []
            for row in week_feature:
                week_encoding.append(np.vstack([row] * aggregation_seconds))
            week_encoding = np.stack(week_encoding)
            week_encoding = rearrange(week_encoding, "n w1 w2 -> n 1 w1 w2")
            features.append(week_encoding)

        features = np.concatenate(features, axis=1)
        all_features.append(features)

    all_features = np.vstack(all_features)
    logger.info(all_features.shape)
    return all_features


def aggregate_sm(
    sm_comm,
    feature_type,
    aggregation_seconds,
    slide_winsize,
    slide_stride,
    f_time,
    f_week,
    f_hol,
    f_fix,
    f_prob,
    f_cumsum,
):
    all_features = []
    for date, sm_df_by_date in tqdm(sm_comm):
        # temp_df shape (t v)
        temp_df = sm_df_by_date.loc[:, 0:3].copy()
        time_range = pd.date_range(
            "00:00:00", periods=24 * 60 * 60 / aggregation_seconds, freq=f"{aggregation_seconds}s"
        ).strftime("%H:%M:%S")
        tile = np.tile(time_range, (aggregation_seconds, 1)).T.reshape(-1)
        tile = pd.Series(tile, name="group", index=temp_df.index)
        temp_df["group"] = tile

        if feature_type == "all":
            # group shape (t1 v)
            grp_mean = temp_df.groupby(["group"]).mean()
            grp_std = temp_df.groupby(["group"]).std()
            grp_max = temp_df.groupby(["group"]).max()
            grp_min = temp_df.groupby(["group"]).min()

            def sad(x):
                def sad_c(x_c):
                    a = np.arange(0, len(x_c))
                    p = np.array(np.meshgrid(a, a)).T.reshape(-1, 2)
                    y = np.abs(x_c.to_numpy()[p].T[0] - x_c.to_numpy()[p].T[1]).sum()
                    return y

                return x.apply(sad_c)

            grp_sad = temp_df.groupby(["group"]).apply(sad)

            def autocor(x):
                def autocor_c(x_c):
                    cor_lag_1 = np.correlate(x_c, x_c, mode="full")[1]
                    return cor_lag_1

                return x.apply(autocor_c)

            grp_autolag1 = temp_df.groupby(["group"]).apply(autocor)
            grp_range = grp_max - grp_min

            features = [grp_min, grp_max, grp_mean, grp_std, grp_sad, grp_autolag1, grp_range]

        elif feature_type == "raw":
            grp_mean = temp_df.groupby(["group"]).mean()
            features = [grp_mean]
        else:
            logger.warning(f"feature type is {feature_type}!!")

        if f_time:
            t = np.arange(24 * 60 * 60)
            t = t / np.max(t)
            df_time = pd.DataFrame(t, columns=["time"], index=temp_df.index)
            features.append(group_feature(df_time, tile, "max"))
        if f_week:
            weekday = date.weekday() / 6
            df_weekday = pd.DataFrame(
                np.array([weekday] * 24 * 60 * 60), columns=["weekday"], index=temp_df.index
            )
            features.append(group_feature(df_weekday, tile, "max"))
        if f_hol:
            swiss_holidays = holidays.Switzerland()
            hol = int(date in swiss_holidays)
            df_hol = pd.DataFrame(
                np.array([hol] * 24 * 60 * 60), columns=["holiday"], index=temp_df.index
            )
            features.append(group_feature(df_hol, tile, "max"))
        if f_fix:
            t1, t2 = timedelta(hours=9).seconds, timedelta(hours=17).seconds
            span_value = zip([(0, t1), (t1, t2), (t2, 24 * 60 * 60)], [1, 0, 1])
            df_fix = pd.DataFrame(span_to_array(span_value), columns=["fix"], index=temp_df.index)
            features.append(group_feature(df_fix, tile, "mean"))
        if f_prob:
            pass
        if f_cumsum:
            df_cumsum = sm_df_by_date.loc[:, 0:3].copy().cumsum()
            features.append(group_feature(df_cumsum, tile, "mean"))

        features = pd.concat(features, axis=1).to_numpy()
        # ((n 60) v) -> (n 60 v)
        features = sliding_window(features, slide_winsize, slide_stride)
        logger.info(features.shape)
        # features = rearrange(features, '(n w) v -> n w v', w=slide_winsize)
        all_features.append(features)

    all_features = np.vstack(all_features)
    return all_features


@buffer_value("joblib", temp_folder, disable=False)
def preprocess(
    household_data,
    threshold,
    feature_type,
    aggregation_seconds,
    slide_winsize,
    slide_stride,
    f_time,
    f_week,
    f_hol,
    f_fix,
    f_prob,
    f_plugs,
    plugs_cum,
    sm_cum,
):
    df_occ = household_data["occupancy"]
    dict_sm = household_data["sm"]
    sm_index = pd.DatetimeIndex(dict_sm.keys())
    intersect_occ_sm_days = df_occ.index.intersection(sm_index)

    df_occ_comm = df_occ.loc[intersect_occ_sm_days]
    sm_comm = [(date, dict_sm[date.strftime("%Y-%m-%d")]) for date in intersect_occ_sm_days]

    agg_appls = []
    if f_plugs:
        for k in household_data:
            if "plugs" in k:
                dict_appl = household_data[k]
                dates = pd.DatetimeIndex(dict_appl.keys())
                appl_dates_inter_occ = intersect_occ_sm_days.intersection(dates)
                if len(appl_dates_inter_occ) == 0:
                    logger.info(f"skip applicances {k} {appl_dates_inter_occ.shape}")
                    continue
                appl_comm = [
                    (date, dict_appl[date.strftime("%Y-%m-%d")])
                    if date.strftime("%Y-%m-%d") in dict_appl
                    else (date, None)
                    for date in intersect_occ_sm_days
                ]
                # agg_appl (n t 1)
                agg_appl = aggregate_appliance(
                    appl_comm, aggregation_seconds, slide_winsize, slide_stride, plugs_cum
                )
                agg_appls.append(agg_appl)
                logger.info(f"appl {k} appl_inter_occ {appl_dates_inter_occ.shape}")
                # plugs_comm = [(date, dict_appl[date.strftime('%Y-%m-%d')]) for date in intersect_occ_sm_days]
                # agg_plugs = aggregate_appliance(df_plugs_comm)

        if agg_appls != []:
            agg_appls = np.concatenate(agg_appls, axis=2)
            agg_appls = rearrange(agg_appls, "n t v -> n 1 v t")
            logger.info(agg_appls.shape)
        # logger.info(f'{k} {index_set.shape}')

    # logger.info(f'intersect {intersect_occ_sm_days.shape}')
    # index_set = index_set.intersection(intersect_occ_sm_days)
    # logger.info(f'index_set {index_set.shape}')

    if feature_type == "cwt":
        # (n 60 v) -> (n 60 60)
        agg_sm = aggregate_sm_cwt(
            sm_comm, aggregation_seconds, slide_winsize, slide_stride, f_time, f_week
        )
    else:
        agg_sm = aggregate_sm(
            sm_comm,
            feature_type,
            aggregation_seconds,
            slide_winsize,
            slide_stride,
            f_time,
            f_week,
            f_hol,
            f_fix,
            f_prob,
            sm_cum,
        )
        agg_sm = rearrange(agg_sm, "n t v -> n 1 v t")

    agg_occ = aggregate_occupancy(
        df_occ_comm, aggregation_seconds, slide_winsize, slide_stride, th=threshold
    )
    logger.debug(f"{agg_occ.shape} {agg_sm.shape}")

    if agg_appls != []:
        logger.info(f"agg_appls.shape {agg_appls.shape}")
        logger.info(f"agg_sm.shape {agg_sm.shape}")
        agg_sm = np.concatenate([agg_sm, agg_appls], axis=2)
        logger.info(agg_sm.shape)

    return agg_sm, agg_occ


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
