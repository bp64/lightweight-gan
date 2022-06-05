import datetime
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_list(el):
    return el if isinstance(el, list) else [el]


def current_iso_datetime():
    return datetime.now().isoformat()
