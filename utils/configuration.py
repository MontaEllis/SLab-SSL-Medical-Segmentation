import os
import logging
import torch
import torch.distributed as dist
import pdb
import yaml
def is_master(args):
    return args.rank % args.ngpus_per_node == 0


def _to_plain_dict(value):
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(v) for v in value]
    return value


def save_configure(args):
    with open(os.path.join(args.experiment.log_path,'config.yaml'), 'w') as f:
        yaml.safe_dump(_to_plain_dict(args), f, sort_keys=False)
