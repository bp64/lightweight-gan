import datetime
import random
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from math import log2

import numpy as np
import torch

from lightweight_gan.exceptions import NanException


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


def is_power_of_two(val):
    return log2(val).is_integer()


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = "" if int(n >= 0) else "-"
        res = float(f"{prefix}inf")
    return res


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


################################################################################
@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]

    return multi_contexts


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


@lru_cache(maxsize=10)
def det_randn(*args):
    """
    deterministic random to track the same latent vars (and images) across training steps
    helps to visualize same image over training steps
    """
    return torch.randn(*args)


def interpolate_between(a, b, *, num_samples, dim):
    assert num_samples > 2
    samples = []
    step_size = 0
    for _ in range(num_samples):
        sample = torch.lerp(a, b, step_size)
        samples.append(sample)
        step_size += 1 / (num_samples - 1)
    return torch.stack(samples, dim=dim)
