#extracted from: https://github.com/kherud/neural-additive-models-pt/tree/master
# required classes and functions for neural additive models
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
from typing import Callable
import pandas as pd
import random
from torch.nn.parameter import Parameter
from typing import Sequence
from typing import Tuple
from torch.nn.parameter import Parameter



def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)
    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)
        
class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)
    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)
        
class FeatureNN(torch.nn.Module):
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], 1, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class NeuralAdditiveModel(torch.nn.Module):
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

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout)
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)

        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]




def calculate_n_units(x_train, n_basis_functions, units_multiplier):
    num_unique_vals = [
        len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])
    ]
    return [
        min(n_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]

# code extracted from https://www.kaggle.com/code/paulo100/tabtransformer-pytorch-dnn-with-attention-eda/notebook:
class Preprocessor(nn.Module):
    def __init__(self, numerical_columns, categorical_columns, encoder_categories, emb_dim):
        super().__init__()
        self.numerical_columns = numerical_columns
        self.numerical_columns.remove("readmitted_binarized")
        self.categorical_columns = categorical_columns
        self.encoder_categories = encoder_categories
        self.emb_dim = emb_dim
        self.embed_layers = nn.ModuleDict()
        
        for i, categorical in enumerate(categorical_columns):
            embedding = nn.Embedding(
                num_embeddings=len(self.encoder_categories[i]),
                embedding_dim=self.emb_dim,
            )
            
            self.embed_layers[categorical] = embedding
        
    def forward(self, x):
    
      x_nums = []
      for numerical in self.numerical_columns:
          x_num = torch.unsqueeze(x[numerical], dim=1)
          x_nums.append(x_num)
      if len(x_nums) > 0:
          x_nums = torch.cat(x_nums, dim=1)
      else:
          x_nums = torch.tensor(x_nums, dtype=torch.float32)
      
      x_cats = []
      for categorical in self.categorical_columns:
          
          x_cat = self.embed_layers[categorical](x[categorical])
          x_cats.append(x_cat)
      if len(x_cats) > 0:
          x_cats = torch.cat(x_cats, dim=1)
      else:
          x_cats = torch.tensor(x_cats, dtype=torch.float32)
      
      return x_nums, x_cats










