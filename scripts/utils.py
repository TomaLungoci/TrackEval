import sys
import os
import argparse
from multiprocessing import freeze_support
from functools import reduce
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

def parse_config(configs):
    config = {}
    for cfg in configs.values():
        config.update(cfg)    
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    filtered_configs = []
    for config_name, config in configs.items():
        filtered_config = {k: v for k, v in config.items() if k in config.keys()}
        filtered_configs.append(filtered_config)
    return tuple(filtered_configs)

# def get_config (evaluator, dataset, metrics):
#     eval_config = trackeval.Evaluator.get_default_eval_config()
#     dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
#     metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
#     return eval_config

# def parse_config (eval_config, dataset_config, metrics_config):
#     config = {**eval_config, **dataset_config, **metrics_config}  # Merge default configs
#     parser = argparse.ArgumentParser()
#     for setting in config.keys():
#         if type(config[setting]) == list or type(config[setting]) == type(None):
#             parser.add_argument("--" + setting, nargs='+')
#         else:
#             parser.add_argument("--" + setting)
#     args = parser.parse_args().__dict__
#     for setting in args.keys():
#         if args[setting] is not None:
#             if type(config[setting]) == type(True):
#                 if args[setting] == 'True':
#                     x = True
#                 elif args[setting] == 'False':
#                     x = False
#                 else:
#                     raise Exception('Command line parameter ' + setting + 'must be True or False')
#             elif type(config[setting]) == type(1):
#                 x = int(args[setting])
#             elif type(args[setting]) == type(None):
#                 x = None
#             else:
#                 x = args[setting]
#             config[setting] = x
#     eval_config = {k: v for k, v in config.items() if k in eval_config.keys()}
#     dataset_config = {k: v for k, v in config.items() if k in dataset_config.keys()}
#     metrics_config = {k: v for k, v in config.items() if k in metrics_config.keys()}
#     return eval_config, dataset_config, metrics_config

def get_all_metrics():
    return [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]

def runner(evaluator, dataset_list, metrics_list, metrics_config, metrics):
    for metric in metrics:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
