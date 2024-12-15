import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
import time


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper


class LossHistory:
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_validate = []
        self.metrics_validate = []

    def append(self, step, loss_train, loss_validate, metrics_validate):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_validate is None:
            loss_validate = self.loss_validate[-1]
        if metrics_validate is None:
            metrics_validate = self.metrics_validate[-1]
        self.loss_validate.append(loss_validate)
        self.metrics_validate.append(metrics_validate)


def list_to_str(nums, precision=2):
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


class TrainingDisplay:
    """Display training progress."""

    def __init__(self):
        self.len_train = 14
        self.len_validate = 14
        self.len_metric = 14
        self.is_header_print = False

    def print_one(self, s1, s2, s3, s4):
        print(
            "{:{l1}s}{:{l2}s}{:{l3}s}{:{l4}s}".format(
                s1,
                s2,
                s3,
                s4,
                l1=10,
                l2=self.len_train,
                l3=self.len_validate,
                l4=self.len_metric,
            )
        )
        sys.stdout.flush()

    def header(self):
        self.print_one("Epoch", "Train loss", "Validate loss", "Validate metric")
        self.is_header_print = True

    def __call__(self, train_state):
        if not self.is_header_print:
            self.header()
        self.print_one(
            str(train_state.epoch),
            list_to_str(train_state.loss_train),
            list_to_str(train_state.loss_validate),
            list_to_str(train_state.metrics_validate),
        )

    def summary(self, train_state):
        print("Best model at step {:d}:".format(train_state.best_step))
        print("  train loss: {:.2e}".format(train_state.best_loss_train))
        print("  validate loss: {:.2e}".format(train_state.best_loss_validate))
        print("  validate metric: {:s}".format(list_to_str(train_state.best_metrics)))
        if train_state.best_ystd is not None:
            print("  Uncertainty:")
            print("    l2: {:g}".format(np.linalg.norm(train_state.best_ystd)))
            print(
                "    l_infinity: {:g}".format(
                    np.linalg.norm(train_state.best_ystd, ord=np.inf)
                )
            )
            print(
                "    max uncertainty location:",
                train_state.X_validate[np.argmax(train_state.best_ystd)],
            )
        print("")
        self.is_header_print = False


class TrainState:
    def __init__(self):
        self.epoch = 0

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Validate results
        self.loss_validate = None
        self.y_pred_validate = None
        self.metrics_validate = None

        # The best results correspond to the min train loss
        self.best_epoch = 0
        self.best_loss_train = np.inf
        self.best_loss_validate = np.inf
        self.best_metrics = None

    def update_train_state(self, epoch, loss_train, loss_validate, metrics_validate):
        self.epoch = epoch
        self.loss_train = loss_train
        self.loss_validate = loss_validate
        self.metrics_validate = metrics_validate
        # 判断是否最优
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_epoch = self.epoch
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_validate = np.sum(self.loss_validate)
            self.best_metrics = self.metrics_validate

    def disregard_best(self):
        self.best_loss_train = np.inf


def get_loss_fun(identifier):
    """获取损失函数"""
    LOSS_DICT = {
        "l1 loss": nn.L1Loss(),
        "mean absolute error": nn.L1Loss(),
        "mae": nn.L1Loss(),
        "mean squared error": nn.MSELoss(),
        "mse": nn.MSELoss(),
    }

    if isinstance(identifier, str):
        return LOSS_DICT[identifier.lower()]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(f'{{get_loss_fun}} Could not interpret loss function {identifier}')


def get_activation(identifier):
    """获取激活函数"""

    def linear(x):
        return x

    if identifier is None:
        # 未指定激活函数返回输入值
        return linear
    elif isinstance(identifier, str):
        identifier = identifier.lower()
        return {
            "elu": F.elu,
            "gelu": F.gelu,
            "relu": F.relu,
            "selu": F.selu,
            "sigmoid": F.sigmoid,
            "silu": F.silu,
            "swish": F.silu,
            "tanh": F.tanh,
        }[identifier.lower()]
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(f'{{get_activation}} Could not interpret {identifier} activation function')
