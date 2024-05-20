import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

from lab3.benchmark.transport import get_stub
from lab3.benchmark.transport.generated.api_pb2 import PredictRequest, DataSet, Row, TrainRequest, TrainerConfig, \
    ModelConfig


def plot3d(objective: tp.Callable):
    # define range for input
    r_min, r_max = -10.0, 10.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)
    # compute targets
    results = objective([x, y])
    # create a surface plot with the jet color scheme
    figure = plt.figure()
    axis = plt.axes(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    axis.set_title(objective.__name__)
    # show the plot
    plt.show()


def plot2d(objective: tp.Callable):
    # Определяем диапазон для входных значений
    r_min, r_max = -10.0, 10.0
    # Генерируем входные значения равномерно с шагом 0.1
    xaxis = np.arange(r_min, r_max, 0.1)
    # Вычисляем значения функции для каждой точки сетки
    results = objective([xaxis])
    # Строим обычный 2D график
    plt.plot(xaxis, results)
    # plt.colorbar(label='Значения функции')
    plt.title(objective.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    # Показываем график
    plt.show()


def _plot_ds(df: pandas.DataFrame):
    columns = df.columns
    num_columns = len(columns)
    fig = plt.figure(dpi=400)

    if num_columns == 2:
        # Если две колонки - одномерный датасет
        param = columns[0]
        value = columns[1]
        ax = plt.axes()
        # Построение графика точек
        ax.scatter(df[param], df[value])
        ax.set_xlabel("param")
        ax.set_ylabel("value")
        ax.set_title(f'Scatter Plot of value vs param')

    elif num_columns == 3:
        x, y, z = df.columns

        # Создание фигуры и 3D осей
        ax = plt.axes(projection='3d')

        # Построение 3D scatter plot
        scatter = ax.scatter(df[x], df[y], df[z], c=df[z], cmap='viridis')

        # Установка меток осей
        ax.set_xlabel("param 1")
        ax.set_ylabel("param 2")
        ax.set_zlabel("value")

        # Добавление цветовой шкалы
        fig.colorbar(scatter, ax=ax, label="value")

        # Заголовок графика
        ax.set_title('3D Scatter Plot')
    else:
        print("Датасет должен содержать либо 2, либо 3 колонки.")
        return
    return ax


def _plot_model(ax, xs: tp.Iterable[float], model_id: str):
    s = get_stub()
    xs = np.linspace(min(xs), max(xs), 1000)
    r = s.predict(PredictRequest(modelId=model_id, data=DataSet(
        rows=[Row(x=(row, )) for row in xs]
    )))
    print(r.y)

    ax.plot(xs, r.y, color='orange')


def plot_dataset(file_path):
    # Чтение CSV файла
    df = pd.read_csv(file_path)

    # Определение количества колонок в датасете
    _plot_ds(df)
    # Отображение графика
    plt.show()


def plot_model_over_dataset(file_path, model_id):
    df = pd.read_csv(file_path)
    ax = _plot_ds(df)
    rows = df.loc[:, df.columns[0]].values
    _plot_model(ax, rows, model_id)
    ax.set_title("Model over dataset")
    plt.show()


if __name__ == '__main__':
    # generate("../impl/test-data/test_1dim.csv", GeneratorConfig(dim=1, rows=1000, noize=2, functions=Functions.LINEAR))
    s = get_stub()
    r = s.train(TrainRequest(
        path="test_1dim.csv",
        trainerConfig=TrainerConfig(type="GreedyTrainer", params=[0.01, 100000, 0.001]),
        modelConfig=ModelConfig(type="PolynomialModel", regularizator="EmptyRegularizator", loss="MSELoss",
                                otherParams=[5]),
    ))
    print(r)
    plot_model_over_dataset("../impl/test-data/test_1dim.csv", r.modelId)
    # plot_model_over_dataset("../impl/test-data/test_1dim.csv", "20284196-2db3-4494-9623-127a58b497de")
