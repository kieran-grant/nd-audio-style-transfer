from typing import List

import torch.nn as nn

from src.spsa.spsa import SPSABatch
from src.wrappers.base_dafx_wrapper import BaseDAFXWrapper


# ==== Wrapper for SPSA layer ====
class DAFXLayer(nn.Module):
    def __init__(self, dafx: BaseDAFXWrapper, epsilon: float):
        super().__init__()
        self.dafx = dafx
        self.epsilon = epsilon
        self.spsa = SPSABatch()

    def forward(self, inputs: List):
        signal = inputs[0]
        params = inputs[1]
        return self.spsa.apply(signal.squeeze(), params.squeeze(), self.dafx, self.epsilon)
