import time
import torch
import torch.distributed as dist
from collections import OrderedDict

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def load_net_state_dict(model, net_state_dict):
    try:
        model.load_state_dict(net_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in net_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
