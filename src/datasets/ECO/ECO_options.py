def get_dataset_setting(setting):
    # fea_type, agg_sec, slide_w, slide_s
    params = {
        1: ["raw", 60, 60, 60],
        2: ["raw", 60, 30, 30],
        3: ["raw", 60, 15, 15],
        4: ["all", 60, 60, 60],
        5: ["all", 60, 30, 30],
        6: ["all", 60, 15, 15],
    }
    p = params[setting]
    p_dict = dict(fea_type=p[0], agg_sec=p[1], slide_w=p[2], slide_s=p[3])
    return p_dict


def get_options():
    options = [
        ["dataset_setting", int, 1],
        ["study_case", str, "case1"],  # study cases: 'case1', 'case2', 'case3', 'case4'
        ["fea_type", str, "raw"],  # 'raw' only use raw power consumption as used in paper
        ["agg_sec", int, 60],  # aggregation seconds
        ["slide_w", int, 60],  # sliding window size
        ["slide_s", int, 60],  # sliding window stripe
        ["norm_type", str, "minmax"],  # feature normalization type
        ["imb_sam", int, 0],  # whether use imbalance sampler
        ["t_en", int, 1],  # encode time of the day
        ["w_en", int, 1],  # encode day of the week
        ["hol_en", int, 0],  # encode holiday
        ["fix_en", int, 0],  #
        ["prob_en", int, 0],  #
        ["plugs_en", int, 0],  # encode plugs data
        ["plugs_cum", int, 0],  # encode cumulated plugs data
        ["sm_cum", int, 0],  # encode smart meter cumulated data
        ["data_aug", str, "RANDOM"],  # data augmentation 'RANDOM' (random oversample) or 'SMOTE'
    ]
    return options
