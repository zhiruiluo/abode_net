import importlib
import inspect
import os

def import_method(module,method_name):
    module = importlib.import_module(f'{module}')
    method = getattr(module, method_name)
    return method

def import_class(module, class_name):
    module = importlib.import_module(f'{module}')
    class_ = getattr(module, class_name)
    return class_

def filter_dict(func, kwarg_dict):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict

def init_class_from_namespace(class_, namespace):
    common_kwargs = filter_dict(class_, vars(namespace))
    return class_(**common_kwargs)

def scan_nn_model_list():
    s = os.path.dirname(os.path.realpath(__file__))
    modules = os.listdir(os.path.join(s,'model'))
    fns = [ f[:-3] for f in modules if not f.endswith('__init__.py') and f.endswith('.py')]
    return fns

class ModelSelection():
    nn_model_list = ['ABODE_Net']
    def __init__(self) -> None:
        pass

    @staticmethod
    def default_model() -> str:
        return ModelSelection.choices()[0]

    @staticmethod
    def choices():
        return ModelSelection.nn_model_list

    @staticmethod
    def getModel(name,args=None):
        if name not in ModelSelection.choices():
            raise ValueError(f'No model named {name}!')
        
        model_class = import_class(f'src.models.{name}',name)
        model = init_class_from_namespace(model_class, args)
        model.args = args

        return model

    @staticmethod
    def getParams(name):
        if name not in ModelSelection.choices():
            raise ValueError(f'No model named {name}!')

        get_options = import_method(f'src.models.{name}', 'get_options')
        options = get_options()
        param_keys = [o[0] for o in options]
        return param_keys, options