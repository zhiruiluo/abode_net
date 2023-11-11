import importlib
import inspect


def import_function_or_class(module, method_name):
    module = importlib.import_module(f"{module}")
    method = getattr(module, method_name)
    return method


def update_dataset_setting(args, get_dataset_setting):
    setting = args.dataset_setting
    param_setting = get_dataset_setting(setting)
    vars(args).update(param_setting)


def filter_dict(func, kwarg_dict):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict


def init_class_from_namespace(class_, namespace):
    common_kwargs = filter_dict(class_, vars(namespace))
    return class_(**common_kwargs)


class DatasetSelection:
    dataset_list = ["ECO", "NIOM"]

    def __init__(self) -> None:
        pass

    @staticmethod
    def default() -> str:
        return DatasetSelection.choices()[0]

    @staticmethod
    def choices():
        return DatasetSelection.dataset_list

    @staticmethod
    def getDataset(name, trainer=None, hpm=None, args=None):
        if name not in DatasetSelection.choices():
            raise ValueError(f"No dataset named {name}")
        if name == "ECO":
            from src.datasets.ECO.ECO import ECODataset
            from src.datasets.ECO.ECO_options import get_dataset_setting

            update_dataset_setting(args, get_dataset_setting)
            dl = init_class_from_namespace(ECODataset, args)
            dl.set_repeation(args.repeat, args.nrepeat)
            dl.prepare_data()
            args.nclass = dl.nclass
            args.max_len = dl.max_len
            args.nc = dl.nc
            args.input_shape = dl.input_shape
            args.variable_len = dl.variable_len
            args.class_weight = dl.train_class_weight
        if name == "NIOM":
            from src.datasets.NIOM.NIOM import NIOMDataset
            from src.datasets.NIOM.NIOM_options import get_dataset_setting

            update_dataset_setting(args, get_dataset_setting)

            dl = init_class_from_namespace(NIOMDataset, args)
            dl.set_repeation(args.repeat, args.nrepeat)
            dl.prepare_data()
            args.nclass = dl.nclass
            args.max_len = dl.max_len
            args.nc = dl.nc
            args.input_shape = dl.input_shape
            args.variable_len = dl.variable_len
            args.class_weight = dl.train_class_weight

        return dl

    @staticmethod
    def getParams(name, args=None):
        if name not in DatasetSelection.choices():
            raise ValueError(f"No dataset named {name}!")

        if name == "ECO":
            from src.datasets.ECO.ECO_options import get_options, get_dataset_setting

            options = get_options()
            if args:
                update_dataset_setting(args, get_dataset_setting)
        if name == "NIOM":
            from src.datasets.NIOM.NIOM_options import get_options, get_dataset_setting

            options = get_options()
            if args:
                update_dataset_setting(args, get_dataset_setting)

        param_keys = [o[0] for o in options]
        return param_keys, options
