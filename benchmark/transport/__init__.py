import grpc

from lab3.benchmark.transport.generated.api_pb2_grpc import MlStub


def __create_stub():
    channel = grpc.insecure_channel("localhost:8888")
    stub = MlStub(channel)
    return lambda: stub


get_stub = __create_stub()
