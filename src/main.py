import argparse
import yaml
import torch
import numpy as np
from collections import OrderedDict

from core.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    model = ModelHandler(config)
    model.train()
    model.test()


def grid_search_main(config):
    print_config(config)
    set_random_seed(config['random_seed'])
    grid_search_hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            grid_search_hyperparams.append(k)


    best_config = None
    best_metric = None
    best_score = -1
    configs = grid(config)
    for cnf in configs:
        print('\n')
        pretrained = True if cnf['out_dir'] is None else False
        for k in grid_search_hyperparams:
            if pretrained:
                cnf['pretrained'] += '_{}_{}'.format(k, cnf[k])
            else:
                cnf['out_dir'] += '_{}_{}'.format(k, cnf[k])
        if pretrained:
            print(cnf['pretrained'])
        else:
            print(cnf['out_dir'])

        model = ModelHandler(cnf)
        dev_metrics = model.train()
        if best_score < dev_metrics[cnf['eary_stop_metric']]:
            best_score = dev_metrics[cnf['eary_stop_metric']]
            best_config = cnf
            best_metric = dev_metrics
            print('Found a better configuration: {}'.format(best_score))

    print('\nBest configuration:')
    for k in grid_search_hyperparams:
        print('{}: {}'.format(k, best_config[k]))

    print('Best score: {}'.format(best_score))

################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        # config = yaml.load(setting, Loader=yaml.FullLoader)
        config = yaml.load(setting)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--grid_search', action='store_true', help='flag: grid search')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]


################################################################################
# Module Command-line Behavior #
################################################################################


if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['grid_search']:
        grid_search_main(config)
    else:
        main(config)
