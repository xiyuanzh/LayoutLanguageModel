# note: Fake Linklink. Just for test offline
import torch


def barrier():
    return


def initialize():
    return


def finalize():
    return


def get_rank():
    return 0


def get_world_size():
    return 1


def synchronize():
    return


def broadcast(data, root):
    return


def allreduce(data):
    return


def allreduce_async(name, data):
    return


class NN:
    SyncBatchNorm2d = torch.nn.BatchNorm2d


nn = NN()
