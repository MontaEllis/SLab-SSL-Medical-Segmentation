from typing import Iterable

import numpy as np


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(args, iteration):
    epoch_proxy = iteration // 150
    rampup = args.method_group.consistency_rampup
    return args.method_group.consistency * sigmoid_rampup(epoch_proxy, rampup)


def adjust_poly_learning_rate(optimizer, base_lr, iteration, max_iterations, power=0.9):
    lr = base_lr * (1.0 - iteration / max_iterations) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_multiple_poly_learning_rates(optimizers: Iterable, base_lr, iteration, max_iterations, power=0.9):
    lr = base_lr * (1.0 - iteration / max_iterations) ** power
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr
