from enum import Enum


class GCNLayerType(Enum):
    conv = 0  # convolution layer
    ln = 1
    acti = 2
    bn = 3


class LossType(Enum):
    clsloss = 0
    rgsloss = 1


class SourceType(Enum):
    inductive = 0
    transductive = 1


class ElementType(Enum):
    gcn = 0
    dataset = 1
    loss = 2


class TrainType(Enum):
    pass
