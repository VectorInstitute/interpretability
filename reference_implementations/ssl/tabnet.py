from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d, ReLU
from typing import Callable
import pandas as pd
import random
from torch.nn.parameter import Parameter
from typing import Sequence
from typing import Tuple
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from transformers import BertModel, LongformerModel,AutoModelForSequenceClassification
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pytorch_tabnet import sparsemax
import scipy
from pytorch_tabnet.tab_network import TabNetPretraining,TabNetPretraining2, TabNet
from dataclasses import dataclass, field
from typing import List, Any, Dict
from pytorch_tabnet.utils import create_group_matrix
import zipfile
import os
# embedder, TabnetEncoder
    
def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class TabNetEncoder2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)
        self.group_attention_matrix = group_attention_matrix

        if self.group_attention_matrix is None:
            # no groups
            self.group_attention_matrix = torch.eye(self.input_dim)
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.attention_dim,
                group_matrix=group_attention_matrix,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
                mask_type=self.mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        bs = x.shape[0]  # batch size
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :]
        steps_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            steps_output.append(d)
            # update attention
            att = out[:, self.n_d :]

        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)
        bs = x.shape[0]  # batch size
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            masks[step] = M_feature_level
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M_feature_level, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d :]

        return M_explain, masks
    
class TabNetPretraining2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        n_shared_decoder=1,
        n_indep_decoder=1,
        group_attention_matrix=None,
    ):
        super(TabNetPretraining2, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio,
                                       group_matrix=self.embedder.embedding_group_matrix)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x):
        """
        Returns: res, embedded_x, obf_vars
            res : output of reconstruction
            embedded_x : embedded input
            obf_vars : which variable where obfuscated
        """
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obfuscated_groups, obfuscated_vars = self.masker(embedded_x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_groups
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obfuscated_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)


class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix
        )

        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


class TabNet2(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=[],
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
            self.embedder.embedding_group_matrix
        )

    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        group_matrix,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        self.bn = GBN(
            group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim, output_dim, first=is_first, **params
            )

    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x

def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)

class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GBN(
            2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out


class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        group_matrix : torch matrix
            Original group matrix before embeddings
        """
        super(EmbeddingGenerator, self).__init__()
        # if group_attention_matrix is None:
            # no groups
            ######################self-added
        self.group_attention_matrix = torch.eye(input_dim)
        self.attention_dim = input_dim
        group_matrix =  self.group_attention_matrix
        # else:
        #     self.attention_dim = self.group_attention_matrix.shape[0]
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.embedding_group_matrix = group_matrix.to(group_matrix.device)
            return
        else:
            self.skip_embedding = False
##############********************

        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        
        cat_emb_dims = [cat_emb_dims for i in sorted_idxs]

######################************
        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        
        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

        # update group matrix
        n_groups = group_matrix.shape[0]
        self.embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim),
                                                  device=group_matrix.device)
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # this means that no embedding is applied to this column
                    self.embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]  # noqa
                    post_emb_idx += 1
                else:
                    # this is a categorical feature which creates multiple embeddings
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    self.embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings  # noqa
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            print("error",len(self.embeddings),cat_feat_counter)
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks.
    The obfuscation is done at group level to match attention.
    """

    def __init__(self, pretraining_ratio, group_matrix):
        """
        This create random obfuscation for self suppervised pretraining
        Parameters
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction

        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio
        # group matrix is set to boolean here to pass all posssible information
        self.group_matrix = (group_matrix > 0) + 0.
        self.num_groups = group_matrix.shape[0]

    def forward(self, x):
        """
        Generate random obfuscation mask.

        Returns
        -------
        masked input and obfuscated variables.
        """
        bs = x.shape[0]

        obfuscated_groups = torch.bernoulli(
            self.pretraining_ratio * torch.ones((bs, self.num_groups), device=x.device)
        )
        obfuscated_vars = torch.matmul(obfuscated_groups, self.group_matrix)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_groups, obfuscated_vars
    
def check_list_groups(list_groups, input_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= input_dim)
        - does not contain empty groups
    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assign a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset
    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = f"One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return

def create_group_matrix(list_groups, input_dim):
    """
    Create the group matrix corresponding to the given list_groups

    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset

    Returns
    -------
    - group_matrix : torch matrix
        A matrix of size (n_groups, input_dim)
        where m_ij represents the importance of feature j in group i
        The rows must some to 1 as each group is equally important a priori.

    """
    check_list_groups(list_groups, input_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix
    
class TabNetPretraining2(torch.nn.Module):
    def __init__(
        self,
        input_dim,cat_idxs,
        cat_dims,
        pretraining_ratio=0.2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        n_shared_decoder=1,
        n_indep_decoder=1,
        group_attention_matrix=None
    ):
        super(TabNetPretraining2, self).__init__()

        self.cat_idxs = cat_idxs 
        self.cat_dims = cat_dims 
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio,
                                       group_matrix=self.embedder.embedding_group_matrix)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )
        
        self.reducing_matrix = create_explain_matrix(
                self.input_dim,
                self.cat_emb_dim,
                self.cat_idxs,
                self.post_embed_dim,
            )
        # self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)
    def forward(self, x):
        """
        Returns: res, embedded_x, obf_vars
            res : output of reconstruction
            embedded_x : embedded input
            obf_vars : which variable where obfuscated
        """
        embedded_x = self.embedder(x)
        
        
        steps_out, _ = self.encoder(embedded_x)
        
        return steps_out, _

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)

#################################################################################################################################################
class BERTModel(nn.Module):
    def __init__(self, hidden_size=256,embedding_dim=128):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output is the [CLS] token representation
        
        return pooled_output


class Longformer(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self,output_dim,dropout=0.25):

        super(Longformer, self).__init__()


    
        self.bert_layer = LongformerModel.from_pretrained("yikuan8/Clinical-Longformer",
                                                        gradient_checkpointing=False,
                                                        attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        
        
        self.embedding_dim: int = 768
        self.output_dim=output_dim
        self.hidden_dim: int = 256
        self.last_n_layers = 1

       
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        for param in self.bert_layer.encoder.layer[11].parameters():
            param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[10].parameters():
        #     param.requires_grad = True
        
        

        self.dropout1 = nn.Dropout(dropout)
        
        



        

    def forward(self, input_ids, attention_mask = None):
        
        # print("****",input_ids.shape)
        
        bert_l=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask)#,return_dict=True,output_attentions=True)
        
        
        
        all_feat = bert_l.last_hidden_state
        
        
        
        report_feat=all_feat[:,0,:]
        
        
        # print("featttt",all_feat.shape,report_feat.shape)
        return  report_feat





class Projector(nn.Module):
    def __init__(self, in_dim=128,out_dim=128):
        super(Projector, self).__init__()
        # in_dim =128# 32
        out_dim = 128#32
        self.dense  =  nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x

class report_ehr_model(nn.Module): 
    def __init__(self, numerical_columns, categorical_columns,
                 num_transformer_blocks, num_heads, emb_dim,
                 attn_dropout_rates, ff_dropout_rates,
                 mlp_dropout_rates,
                 mlp_hidden_units_factors,
                 ):
        super().__init__()
        self.report_transformer = Longformer(output_dim=64)
        self.transformers = nn.Sequential()
        # self.report_transformer = BERTModel()
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
        # self.test_layer = nn.Linear(1,12)
        # self.final_dense = nn.Linear(mlp_hidden_units[-1], 1)
        self.final_sigmoid = nn.Sigmoid()
        self.report_projector = Projector(768)
        self.ehr_projector = Projector(128)
    def forward(self, input_ids,attention_mask,x_nums, x_cats):
        # print("cattttttt",x_nums.shape)
        # print("finallllll",x_nums.shape,x_cats.shape)
        
        contextualized_x_cats = self.transformers(x_cats)
        contextualized_x_cats = self.flatten(contextualized_x_cats)
        
        if x_nums.shape[-1] > 0:
            x_nums = self.num_norm(x_nums)
            # x_nums = x_nums.unsqueeze(-1)
            # x_nums = self.test_layer(x_nums)
            # print("cat_num",x_nums.shape,contextualized_x_cats.shape)
            # x_nums = self.flatten(x_nums)
            features = torch.cat((x_nums, contextualized_x_cats), -1)

        else:
            features = contextualized_x_cats
            
        ehr_embedding = self.mlp(features)
        # print("ehrrrr",ehr_embedding.shape)
        # ehr_embedding = self.final_dense(mlp_output)
        # output = self.final_sigmoid(model_output)
        report_embedding = self.report_transformer(input_ids,attention_mask)
        report_embedding = self.report_projector(report_embedding)
        ehr_embedding = self.ehr_projector(ehr_embedding)
        # print("embeddings",report_embedding.shape,ehr_embedding.shape)
        return report_embedding,ehr_embedding








class report_ehr_model_combined(nn.Module): 
    def __init__(self, num_cat_cols, num_num_cols, cat_idxs,
                 cat_dims
                 ):
        super().__init__()
        self.num_num_cols = num_num_cols
        self.num_cat_cols = num_cat_cols
        input_dim = self.num_num_cols + self.num_cat_cols
        
        self.report_transformer = Longformer(output_dim=64)
        
        self.grouped_features : List[List[int]] = field(default_factory=list)
        
        unsupervised_model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-4),
        mask_type='entmax', # "sparsemax",
        n_shared_decoder=1, # nb shared glu for decoding
        n_indep_decoder=1, # nb independent glu for decoding
    #     grouped_features=[[0, 1]], # you can group features together here
        
        verbose=5,
    )
        unsupervised_model.input_dim=input_dim
        print("yyyyy",unsupervised_model.input_dim)
        unsupervised_model._set_network()
        # print("ddddd",dir(unsupervised_model))
        self.ehr_encoder = unsupervised_model.network
        del self.ehr_encoder.decoder
        del self.ehr_encoder.masker
        # print("heyyyy",unsupervised_model)
        net_dict = self.ehr_encoder.state_dict()
        weight_path = './tabnet_pretrained_new_dim_2.zip'

        # with zipfile.ZipFile(weight_path, 'r') as zip_ref:
        #     zip_ref.extractall("output_weight")
        pretrain = torch.load(os.path.join("output_weight",'network.pt'))
        # print("firsttttt",pretrain.items(),net_dict.keys())
        pretrain_cnn={k: v for k, v in pretrain.items() if k in net_dict.keys()}
        # pretrain_cnn = {k.replace('cnn.', ''): v for k, v in pretrain_cnn.items()}
        # print("modellllllllllll",pretrain_cnn.keys())
        self.ehr_encoder.load_state_dict(pretrain_cnn)#***

        for pname, p in self.ehr_encoder.embedder.named_parameters():
                p.requires_grad = False
        for param in self.ehr_encoder.encoder.initial_bn.parameters():
            param.requires_grad = False
        for param in self.ehr_encoder.encoder.initial_splitter.parameters():
            param.requires_grad = False
        # for param in self.ehr_encoder.encoder.feat_transformers[:].parameters():
        #     param.requires_grad = False
        # for param in self.ehr_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.ehr_encoder.encoder.feat_transformers[2].shared.glu_layers.parameters():
        #     param.requires_grad = False
        # ehr_encoder.encoder.feat_transformers[2]shared.glu_layers
        # print("ehr_encoder",self.ehr_encoder)
        

        def forward2(self, x):
            
            embedded_x = self.embedder(x)
            steps_out, _ = self.encoder(embedded_x)
            return steps_out, embedded_x, torch.ones(embedded_x.shape).to(x.device)

        
        
        self.final_sigmoid = nn.Sigmoid() 
        self.report_projector = Projector(in_dim=768)
        self.ehr_projector = Projector(128)
        
        

        
        
       
    def forward(self, input_ids,attention_mask,df):
        
        
        
        

        ehr_embedding,_ = self.ehr_encoder(df)
       
        
        report_embedding = self.report_transformer(input_ids,attention_mask)
        report_embedding = self.report_projector(report_embedding)
        ehr_embedding = self.ehr_projector(ehr_embedding)
        # print("embeddings",report_embedding.shape,ehr_embedding.shape)
        return report_embedding,ehr_embedding 

    


