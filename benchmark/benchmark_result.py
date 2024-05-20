import dataclasses
import os
from collections import defaultdict
from dataclasses import dataclass
import typing as tp

import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing_extensions import overload

from common.utils import *
from lab3.benchmark.transport import get_stub
from lab3.benchmark.transport.generated.api_pb2 import TrainResponse, TrainRequest, PredictRequest, DataSet, Row

MAX_INT = 10 ** 9


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    memory: int
    time: float

    def __iter__(self):
        return iter(dataclasses.asdict(self).values())

    def headers(self):
        return list(dataclasses.asdict(self).keys())

    def parameters(self):
        return list(filter(lambda el: el != 'name', dataclasses.asdict(self).keys()))


Runnable = tp.Callable[[], ExperimentResult]


class BenchmarkResult:
    results: tp.List['ExperimentResult']

    # noinspection PyTypeChecker

    def __init__(self, results: tp.List['ExperimentResult']):
        self.results = results

    def all(self):
        if len(self.results) == 0:
            return None
        return self.results[0].parameters()

    @classmethod
    def compare(cls, runnables: tp.Iterable[Runnable]):
        res = []
        for r in runnables:
            print(f"--running {r.__name__}")
            res.append(r())
        return BenchmarkResult(res)

    def print_results(self, sort_by: tp.Callable = None, name="RESULTS"):
        print(name)
        res = sorted(self.results, key=sort_by) if sort_by is not None else self.results
        print(tabulate(res, headers=self.results[0].headers(), tablefmt='orgtbl',
                       floatfmt=".8f"))
        print()

    def __repr__(self):
        return str(self.results)

    def top(self, *fields: str, silent: bool = False, total=False):
        score = defaultdict(int)
        for f in fields:
            sort_by = lambda r: getattr(r, f)
            for i, res in enumerate(sorted(self.results, key=sort_by)):
                score[res.name] += i + 1
            if not silent:
                self.print_results(sort_by=sort_by, name=f"sorted by {f}")

        if not silent and total:
            print()
            print("total results (less score means more efficient)")
            fmt = []
            for it in sorted(score.items(), key=lambda el: el[1]):
                fmt.append(it)
            print(tabulate(fmt, headers=("name", "score"), tablefmt='orgtbl'))

        if silent:
            return score

    @classmethod
    def series(cls, runnables: tp.Iterable[tp.Iterable[Runnable]]) \
            -> tp.List['BenchmarkResult']:
        res = []
        for i, rs in enumerate(runnables):
            print(f"running experiment {i + 1}")
            res.append(cls.compare(rs))
        return res

    @classmethod
    def plot_results(cls, results: tp.List['BenchmarkResult'], fields: tp.Tuple, names: tp.List[str]):
        assert len(results) > 0
        assert len(fields) < 10
        runner_names = list(map(lambda el: el.name, results[0].results))
        colors = list(mcolors.TABLEAU_COLORS.values())
        assert len(runner_names) <= len(colors)
        plt.figure(figsize=(8, 4 * len(fields)), dpi=300)
        for i, f in enumerate(fields):
            fig = plt.subplot(int(f"{len(fields)}1{i + 1}"))
            fig.set_title(f)

            for j, runner in enumerate(runner_names):
                values = []

                for r in results:
                    if f == 'top':
                        values.append(r.top(*r.results[0].parameters(), silent=True)[runner])
                    else:
                        values.append(getattr(r.results[j], f))

                plt.plot(names, values, color=colors[j], label=runner)
            if i == 0:
                plt.legend()

        plt.show()


def train_and_test(name: str, test_dataset_path: str, trainRequest: TrainRequest) -> ExperimentResult:
    print(f"----{name}")
    s = get_stub()
    train_response: TrainResponse = s.train(trainRequest)
    df = pd.read_csv(test_dataset_path)
    xs = df.loc[:, df.columns[:-1]].values
    real_values = df.loc[:, df.columns[-1]].values
    predict_response = s.predict(PredictRequest(modelId=train_response.modelId, data=DataSet(
        rows=[Row(x=row) for row in xs]
    )))
    accuracy = calc_accuracy(real_values, predict_response.y)
    res = ExperimentResult(
        name=name,
        accuracy=accuracy,
        memory=train_response.benchmark.mem,
        time=train_response.benchmark.time / 1000 / 1000
    )
    return res


def calc_accuracy(real_values, predicted_values):
    score = 1
    for predicted, real in zip(predicted_values, real_values):
        score += (predicted - real) ** 2
    return 1 / score


if __name__ == '__main__':
    fs = []
    for i in range(10):
        def f(i):
            return lambda: ExperimentResult(f"test{i}", 0.1 * i, 10 * i, 10000 / (i + 1))


        fs.append(f(i))
    br = BenchmarkResult.compare(fs)
    br.top("accuracy")
