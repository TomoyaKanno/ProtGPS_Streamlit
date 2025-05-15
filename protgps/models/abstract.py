import torch.nn as nn
from protgps.utils.classes import ProtGPS
from abc import ABCMeta, abstractmethod

# from efficientnet_pytorch import EfficientNet
import math


class AbstractModel(nn.Module, ProtGPS):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(AbstractModel, self).__init__()
