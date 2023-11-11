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
        ["study_case", str, "case1"],  # study cases: 'case1', 'case2', 'case3',
        ["fea_type", str, "raw"],
        ["agg_sec", int, 60],
        ["slide_w", int, 60],
        ["slide_s", int, 60],
        ["norm_type", str, "minmax"],
        ["imb_sam", int, 0],
        ["t_en", int, 1],
        ["w_en", int, 1],
        ["hol_en", int, 0],
        ["data_aug", str, "RANDOM"],
    ]
    return options
