import dataclasses
import functools
import math
import os.path
import random

import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import pandas
import pandas as pd

from lab3.benchmark.visualization import plot_dataset
from transport.generated.api_pb2 import Model

DATASET_ROOT = "../impl/test-data"
def full_path(p):
    return os.path.join(DATASET_ROOT, p)

class Functions:
    LINEAR = [
        lambda x: x,
        lambda x: 1
    ]
    ALL = [
        np.cos, np.sin,
        *[(lambda x: x ** p) for p in range(2)],
        *LINEAR
    ]


@dataclasses.dataclass
class GeneratorConfig:
    dim: int = 2
    rows: int = 10_000
    noize: float = 0.01
    dist_y: float = 5
    dist_x: float = 5
    split: tp.Tuple[float, float] = (1, 0)
    functions: tp.List[tp.Callable[[float], float]] = dataclasses.field(default_factory=lambda: Functions.ALL)


@dataclasses.dataclass
class Row:
    x: np.ndarray
    y: float


def generate(output: str, cfg: GeneratorConfig):
    train_out = output if cfg.split[1] == 0 else f"train.{output}"
    assert 0 <= cfg.split[0] <= 1
    assert 0 <= cfg.split[1] <= 1
    assert cfg.split[1] + cfg.split[0] == 1

    open(full_path(train_out), 'w').close()
    batch_size = 10_000
    noize = np.random.uniform(-cfg.noize, cfg.noize, (cfg.rows + 1,))
    raw_func = _generate_function(cfg.dim, cfg.functions)

    def _generate(output, amount):
        i = 0
        buffer = [None] * batch_size
        while i < amount:
            idx = i % batch_size
            if idx == 0:
                _save_buffer(output, buffer)
            x = (np.random.random(cfg.dim) - 0.5) * cfg.dist_x
            buffer[idx] = [*x, raw_func(x * cfg.dist_y) + noize[idx]]
            i += 1
        _save_buffer(output, buffer)

    _generate(train_out, cfg.rows * cfg.split[0])
    test_out = train_out
    if cfg.split[1] > 0:
        test_out = f"test.{output}"
        open(full_path(test_out), 'w').close()
        _generate(test_out, cfg.rows * cfg.split[1])

    return raw_func, train_out, test_out


def _save_buffer(path: str, buffer: tp.List[tp.List[float]]):
    with open(full_path(path), 'a') as f:
        lines = []
        for row in filter(lambda el: el is not None, buffer):
            lines.append(",".join(map(str, row)) + "\n")
        f.writelines(lines)


def _generate_function(dim: int, functions: list[tp.Callable[[float], float]]) -> tp.Callable[[np.ndarray], float]:
    border = 5
    coefs = [(np.random.random(len(functions)) - 0.5) * border for _ in range(dim)]

    def inner(vec: np.ndarray) -> int:
        res = 0
        for var in range(dim):
            for i, f in enumerate(functions):
                res += f(vec[var]) * coefs[var][i]
        return res

    return inner


if __name__ == '__main__':
    pass
    # plot3d(_generate_function(2, Functions.LINEAR))
    # plot2d(_generate_function(1, [*Functions.LINEAR, lambda x: x ** 2]))
    # generate("test_1dim.csv", GeneratorConfig(dim=1, rows=2000, noize=2, functions=Functions.LINEAR))
    # plot_dataset("test_1dim.csv")
    # plot_csv("test_1dim.csv")
