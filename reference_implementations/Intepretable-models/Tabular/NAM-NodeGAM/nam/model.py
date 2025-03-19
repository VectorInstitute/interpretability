#extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

from .utils import truncated_normal_

class ActivationLayer(nn.Module):
    """
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("Abstract method called")

class ExULayer(ActivationLayer):
    """
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        self._init()

    def _init(self):
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)

class ReLULayer(ActivationLayer):
    """
    """
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)

class FeatureNN(nn.Module):
    """
    """
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], 
                             hidden_units[i])
                for i in range(len(hidden_units))
            ]
        )
        self.layers.insert(0, shallow_layer(1, shallow_units))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1],
                                1, bias=False)
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)

class NeuralAdditiveModel(nn.Module):
    """
    """
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = input_size
        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]
        self.feature_nns = torch.nn.ModuleList(
            [
                FeatureNN(shallow_units=shallow_units[i],
                          hidden_units=hidden_units,
                          shallow_layer=shallow_layer,
                          hidden_layer=hidden_layer,
                          dropout=hidden_dropout)
                for i in range(input_size)
            ]
        )
        self.feature_dropout = nn.Dropout(p=feature_dropout)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)

        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]