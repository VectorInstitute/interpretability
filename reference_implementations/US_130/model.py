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
    # print("eyyyy",x_train.shape)
    # print(len(np.unique(x_train[:, 0])))
    num_unique_vals = [
        len(np.unique(x_train[:, i])) for i in range(x_train.shape[1])
    ]
    # print("herrreeeeeeeeeeeee",num_unique_vals)
    return [
        min(n_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]








class TabNet(nn.Module):
    def __init__(self,num_features):
        super(TabNet, self).__init__()
        
        self.num_features = num_features
        self.SA = SelfAttention(self.num_features)
        self.fc1 = nn.Linear(self.num_features, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.SA(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



class SelfAttention(nn.Module): # Parts of this class were extracted from: https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        # Define the key, query, and value linear transformations

        # self.embed_size = embed_size
        # self.heads = heads
        # self.head_dim = embed_size // heads
        self.embed_size = in_channels
        self.key_conv = nn.Linear(in_channels, in_channels, bias=False)#, kernel_size=1)
        self.query_conv = nn.Linear(in_channels, in_channels, bias=False)#, kernel_size=1)
        self.value_conv = nn.Linear(in_channels, in_channels, bias=False)#, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Attention softmax
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.2)
        

    def forward(self, x):
       
        N = x.shape[0]
        value_len = x.shape[1]
        # print("xxxx",x.shape)
        # values = values.reshape(N, value_len, self.heads, self.head_dim)
        
        key = self.key_conv(x)
        # key = self.dropout(key)
        
        query = self.query_conv(x)
        # query = self.dropout(query)
        
        value = self.value_conv(x)
        # value = self.dropout(value)
        
        print("sizeee", x.size(),key.shape)
        # batch_size, ,  = x.size()

        query = query.view(batch_size, channels,-1)#,depth, -1)
        
        key = key.view(batch_size, channels,-1)#,depth, -1)
        
        value = value.reshape(batch_size, channels,-1)#*depth, -1)
        
        
        attention = torch.bmm(query.permute(0, 2, 1), key)
        
        attention = self.softmax(attention)

        
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # print("selfffffffffff",out.shape)
        
        out = out.view(batch_size, channels,height, width, depth)#  depth,height, width).permute(0,1,3,4,2)

        
        temp = out
        # Residual connection and scaling
        out = self.gamma * out+ x
        
        out = out.view(batch_size,channels,  height, width,depth)
        
        
        return out#,attention


class ExU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self.init_params()

    
    def init_params(self):
        self.weight = nn.init.normal_(self.weight, mean=4., std=.5)
        self.bias = nn.init.normal_(self.bias, std=.5)

    
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight))
        out = torch.clamp(out, 0, 1)
        return out


class ReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_params()


    def init_params(self):        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.linear.bias, std=.5)


    def forward(self, x):
        out = self.linear(x)
        out = F.relu(out)
        return out



class FeatureNet(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate = .2, use_exu = True):
        super(FeatureNet, self).__init__()
        layers = [
            ExU(1, hidden_sizes[0]) if use_exu else ReLU(1, hidden_sizes[0])
        ]
        input_size = hidden_sizes[0]
        for s in hidden_sizes[1:]:
            layers.append(ReLU(input_size, s))
            layers.append(nn.Dropout(dropout_rate))
            input_size = s
        layers.append(nn.Linear(input_size, 1, bias = False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NAM_2(nn.Module):
    def __init__(self, no_features, hidden_sizes, dropout_rate = .2, feature_dropout = 0.0, use_exu = True):
        super(NAM_2, self).__init__()
        self.no_features = no_features
        feature_nets = [FeatureNet(hidden_sizes, dropout_rate, use_exu) for _ in range(no_features)]
        self.feature_nets = nn.ModuleList(feature_nets)
        self.feature_drop = nn.Dropout(feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1,), requires_grad=True)
            
    def forward(self, x):
        y = []
        for i in range(self.no_features):
            o = self.feature_nets[i](x[:,i].unsqueeze(1))
            y.append(o)
        y = torch.cat(y, 1)
        y = self.feature_drop(y)
        out = torch.sum(y, axis = -1) + self.bias
        out = torch.sigmoid(out)
        return out, y


# code extracted from https://www.kaggle.com/code/paulo100/tabtransformer-pytorch-dnn-with-attention-eda/notebook:
class Preprocessor(nn.Module):
    def __init__(self, numerical_columns, categorical_columns, encoder_categories, emb_dim):
        super().__init__()
        self.numerical_columns = numerical_columns
        print("nummm",numerical_columns)
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

class MLPBlock(nn.Module):
    def __init__(self, n_features, hidden_units,
                 dropout_rates):
        super().__init__()
        self.mlp_layers = nn.Sequential()
        num_features = n_features
        for i, units in enumerate(hidden_units):
            self.mlp_layers.add_module(f'norm_{i}', nn.BatchNorm1d(num_features))
            self.mlp_layers.add_module(f'dense_{i}', nn.Linear(num_features, units))
            self.mlp_layers.add_module(f'act_{i}', nn.SELU())
            self.mlp_layers.add_module(f'dropout_{i}', nn.Dropout(dropout_rates[i]))
            num_features = units
    def forward(self, x):
        y = self.mlp_layers(x)
        return y


class TabTransformerBlock(nn.Module):
    def __init__(self, num_heads, emb_dim,
                 attn_dropout_rate, ff_dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads,
                                          dropout=attn_dropout_rate,
                                          batch_first=True)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Dropout(ff_dropout_rate), 
            nn.Linear(emb_dim*4, emb_dim))
    def forward(self, x_cat):
        attn_output, attn_output_weights = self.attn(x_cat, x_cat, x_cat)
        x_skip_1 = x_cat + attn_output
        x_skip_1 = self.norm_1(x_skip_1)
        feedforward_output = self.feedforward(x_skip_1)
        x_skip_2 = x_skip_1 + feedforward_output
        x_skip_2 = self.norm_2(x_skip_2)
        return x_skip_2


class TabTransformer(nn.Module): 
    def __init__(self, numerical_columns, categorical_columns,
                 num_transformer_blocks, num_heads, emb_dim,
                 attn_dropout_rates, ff_dropout_rates,
                 mlp_dropout_rates,
                 mlp_hidden_units_factors,
                 ):
        super().__init__()
        self.transformers = nn.Sequential()
        for i in range(num_transformer_blocks):
            self.transformers.add_module(f'transformer_{i}', 
                                        TabTransformerBlock(num_heads,
                                                            emb_dim,
                                                            attn_dropout_rates[i],
                                                            ff_dropout_rates[i]))
        
        self.flatten = nn.Flatten()
        self.num_norm = nn.LayerNorm(len(numerical_columns))
        
        self.n_features = (len(categorical_columns) * emb_dim) + len(numerical_columns)
        mlp_hidden_units = [int(factor * self.n_features) \
                            for factor in mlp_hidden_units_factors]
        self.mlp = MLPBlock(self.n_features, mlp_hidden_units,
                            mlp_dropout_rates)
        
        self.final_dense = nn.Linear(mlp_hidden_units[-1], 1)
        self.final_sigmoid = nn.Sigmoid()
    def forward(self, x_nums, x_cats):
        contextualized_x_cats = self.transformers(x_cats)
        contextualized_x_cats = self.flatten(contextualized_x_cats)
        
        if x_nums.shape[-1] > 0:
            x_nums = self.num_norm(x_nums)
            features = torch.cat((x_nums, contextualized_x_cats), -1)

        else:
            features = contextualized_x_cats
            
        mlp_output = self.mlp(features)
        model_output = self.final_dense(mlp_output)
        # output = self.final_sigmoid(model_output)
        return model_output


class Model(torch.nn.Module):

    def __init__(self, config, name):
        super(Model, self).__init__()
        self._config = config
        self._name = name

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._name


class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

class NAM(Model):

    def __init__(
        self,
        config,
        name,
        *,
        num_inputs: int,
        num_units: int,
    ) -> None:
        super(NAM, self).__init__(config, name)

        self._num_inputs = num_inputs
        self.dropout = nn.Dropout(p=self.config.dropout)

        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(config=config, name=f'FeatureNN_{i}', input_shape=1, num_units=self._num_units[i], feature_num=i)
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self._num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out
















