from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainRequest(_message.Message):
    __slots__ = ("path", "trainerConfig", "modelConfig")
    PATH_FIELD_NUMBER: _ClassVar[int]
    TRAINERCONFIG_FIELD_NUMBER: _ClassVar[int]
    MODELCONFIG_FIELD_NUMBER: _ClassVar[int]
    path: str
    trainerConfig: TrainerConfig
    modelConfig: ModelConfig
    def __init__(self, path: _Optional[str] = ..., trainerConfig: _Optional[_Union[TrainerConfig, _Mapping]] = ..., modelConfig: _Optional[_Union[ModelConfig, _Mapping]] = ...) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = ("type", "regularizator", "loss", "otherParams")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REGULARIZATOR_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    OTHERPARAMS_FIELD_NUMBER: _ClassVar[int]
    type: str
    regularizator: str
    loss: str
    otherParams: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, type: _Optional[str] = ..., regularizator: _Optional[str] = ..., loss: _Optional[str] = ..., otherParams: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainerConfig(_message.Message):
    __slots__ = ("type", "params")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    type: str
    params: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, type: _Optional[str] = ..., params: _Optional[_Iterable[float]] = ...) -> None: ...

class DataSet(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Row]
    def __init__(self, rows: _Optional[_Iterable[_Union[Row, _Mapping]]] = ...) -> None: ...

class Benchmark(_message.Message):
    __slots__ = ("time", "mem")
    TIME_FIELD_NUMBER: _ClassVar[int]
    MEM_FIELD_NUMBER: _ClassVar[int]
    time: int
    mem: int
    def __init__(self, time: _Optional[int] = ..., mem: _Optional[int] = ...) -> None: ...

class Row(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: float
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[float] = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("type", "weights", "bias")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    BIAS_FIELD_NUMBER: _ClassVar[int]
    type: str
    weights: _containers.RepeatedScalarFieldContainer[float]
    bias: float
    def __init__(self, type: _Optional[str] = ..., weights: _Optional[_Iterable[float]] = ..., bias: _Optional[float] = ...) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("modelId", "benchmark")
    MODELID_FIELD_NUMBER: _ClassVar[int]
    BENCHMARK_FIELD_NUMBER: _ClassVar[int]
    modelId: str
    benchmark: Benchmark
    def __init__(self, modelId: _Optional[str] = ..., benchmark: _Optional[_Union[Benchmark, _Mapping]] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("modelId", "data")
    MODELID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    modelId: str
    data: DataSet
    def __init__(self, modelId: _Optional[str] = ..., data: _Optional[_Union[DataSet, _Mapping]] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("y", "benchmark")
    Y_FIELD_NUMBER: _ClassVar[int]
    BENCHMARK_FIELD_NUMBER: _ClassVar[int]
    y: _containers.RepeatedScalarFieldContainer[float]
    benchmark: Benchmark
    def __init__(self, y: _Optional[_Iterable[float]] = ..., benchmark: _Optional[_Union[Benchmark, _Mapping]] = ...) -> None: ...

class GetModelRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...
