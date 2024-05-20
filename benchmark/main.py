# generate("../impl/test-data/test_1dim.csv", GeneratorConfig(dim=1, rows=200, noize=2, functions=Functions.LINEAR))
# s = get_stub()
# r = s.train(TrainRequest(
#     path="./test-data/test_1dim.csv",
#     trainerConfig=TrainerConfig(type="GreedyTrainer", params=[0.01, 100000, 0.001]),
#     modelConfig=ModelConfig(type="LinearModel", regularizator="EmptyRegularizator", loss="MSELoss"),
# ))
# print(r)
# plot_model_over_dataset("../impl/test-data/test_1dim.csv", r.modelId)
# r = s.getModel(GetModelRequest(id="d3497d95-1040-41a2-a511-e1169e60683a"))
# print(r)
# r = s.predict(PredictRequest(modelId="d1c446b1-b994-4b19-a3fe-751812d73576", data=DataSet(
#     rows=[
#         Row(x=[-0.29451304412658563, -0.49304650406076056]),
#         *[Row(x=[-0.3207816957162245, 0.4676085387753295])] * 100000,
#     ]
# )))
# print(r)

from lab3.benchmark.benchmark_result import train_and_test, BenchmarkResult
from lab3.benchmark.dataset_generator import generate, GeneratorConfig, Functions
from transport.generated.api_pb2 import TrainRequest, TrainerConfig, ModelConfig
from transport import get_stub
from visualization import plot_model_over_dataset


def main1():
    name = "deg_1dim.csv"
    _, train, test = generate(name,
                              GeneratorConfig(dim=1, rows=200, noize=2, functions=Functions.ALL, split=(0.7, 0.3)))
    names = []

    def experiment(idx):
        deg = idx
        name = f"deg-{deg}"
        names.append(name)
        f = lambda: train_and_test(name, f"../impl/test-data/{test}", TrainRequest(
            path=train,
            trainerConfig=TrainerConfig(type="GreedyTrainer", params=[0.01, 100000, 0.001]),
            modelConfig=ModelConfig(type="PolynomialModel", regularizator="EmptyRegularizator", loss="MSELoss",
                                    otherParams=[deg]),
        ))
        f.__name__ = name
        return f

    res = []
    for i in range(2, 10):
        res.append(experiment(i))

    br = BenchmarkResult.compare(res)
    br.top(*br.results[0].parameters())


def main2():
    name = "dataset_1dim.csv"
    _, train, test = generate(name,
                              GeneratorConfig(dim=1, rows=1000, noize=2, functions=Functions.ALL, split=(0.7, 0.3)))
    s = get_stub()
    r = s.train(TrainRequest(
        path=test,
        trainerConfig=TrainerConfig(type="GreedyTrainer", params=[0.01, 100000, 0.001]),
        modelConfig=ModelConfig(type="PolynomialModel", regularizator="EmptyRegularizator", loss="MSELoss",
                                otherParams=[5]),
    ))
    print(r)
    plot_model_over_dataset(f"../impl/test-data/{name}", r.modelId)
    # plot_model_over_dataset(f"../impl/test-data/{name}", "f22943a8-8731-4e7a-aa17-f2a43c2f5f1b")


if __name__ == '__main__':
    # main2()
    main1()
