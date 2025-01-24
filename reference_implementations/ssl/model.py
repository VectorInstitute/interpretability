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
import matplotlib.pyplot as plt
from transformers import BertModel, LongformerModel,AutoModelForSequenceClassification
from pytorch_tabnet.pretraining import TabNetPretrainer

class Preprocessor(nn.Module):
    def __init__(self, emb_dim,encoder_categories,categorical_columns):
        super().__init__()
        # self.numerical_columns = numerical_columns
        # print("nummm",numerical_columns)
        
        # self.categorical_columns = categorical_columns
        # if "outcome_critical" in self.numerical_columns:
        #     self.numerical_columns.remove("outcome_critical")
        # elif "outcome_critical" in self.categorical_columns:
        #     self.categorical_columns.remove("outcome_critical")
        # self.encoder_categories = encoder_categories
        self.emb_dim = emb_dim
        self.encoder_categories = encoder_categories

        self.embed_layers = nn.ModuleDict()
        
        for i, categorical in enumerate(categorical_columns):
            embedding = nn.Embedding(
                num_embeddings=len(self.encoder_categories[i]),
                embedding_dim=self.emb_dim,
            )
            
            self.embed_layers[categorical] = embedding
        
    def forward(self,df, numerical_columns, categorical_columns):
     
        # self.embed_layers = nn.ModuleDict()
        
        # for i, categorical in enumerate(categorical_columns):
        #     embedding = nn.Embedding(
        #         num_embeddings=len(self.encoder_categories[i]),
        #         embedding_dim=self.emb_dim,
        #     )
            
        #     self.embed_layers[categorical] = embedding
        x_nums = []
        for numerical in numerical_columns:
            x_num = torch.unsqueeze(df[numerical], dim=1)
            x_nums.append(x_num)
        if len(x_nums) > 0:
            x_nums = torch.cat(x_nums, dim=1)
        else:
            x_nums = torch.tensor(x_nums, dtype=torch.float32)
        
        x_cats = []
        for categorical in categorical_columns:
            
            x_cat = self.embed_layers[categorical](df[categorical])
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
        self.num_features = n_features
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        # self.instance_norm = nn.InstanceNorm1d(self.num_features)
        for i, units in enumerate(hidden_units):
            # print("salammmmmm",i,units,dropout_rates[i])
            # self.mlp_layers.add_module(f'norm_{i}', nn.BatchNorm1d(num_features))
            self.mlp_layers.add_module(f'dense_{i}', nn.Linear(1,units))#(num_features, units))
            self.mlp_layers.add_module(f'act_{i}', nn.SELU())
            self.mlp_layers.add_module(f'dropout_{i}', nn.Dropout(dropout_rates[i]))
            num_features = units

        self.dense1 =  nn.Linear(852,128)#(434,512)
        self.act =  nn.SELU()
        # self.dense2 = nn.Linear()#(512,256)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.1)
    def forward(self, x):
        
        hidden_units = 434
        # try:
        x = self.batch_norm(x)
        # except:
        #     x = self.instance_norm(x)
        # x = x.unsqueeze(-1)
        # print("xxxxx",x.shape)
        x = self.dense1(x)
        x = self.act(x)
        x = self.drop1(x)
        # print("xxxxx1",x.shape)
        # x = self.dense2(x)
        # x = self.act(x)
        # x = self.drop2(x)
        # print("xxxxx2",x.shape)
        # y = self.mlp_layers(x)


        # hidden_units = 217


        return x


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
        attn_output, _ = self.attn(x_cat, x_cat, x_cat)
        # print("atttennn shapes",attn_output.shape,attn_output_weights.shape)
        x_skip_1 = x_cat + attn_output
        x_skip_1 = self.norm_1(x_skip_1)
        feedforward_output = self.feedforward(x_skip_1)
        x_skip_2 = x_skip_1 + feedforward_output
        x_skip_2 = self.norm_2(x_skip_1)
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
        # print("cattttttt",x_nums.shape)
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
    

class BERTModel(nn.Module):
    def __init__(self, hidden_size=256,embedding_dim=128):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(embedding_dim, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output is the [CLS] token representation
        # custom_embedding = self.fc(pooled_output)
        # custom_embedding = self.relu(custom_embedding)


        # logits = self.classifier(custom_embedding)
        return pooled_output


class Longformer(nn.Module): #(GPT2ForSequenceClassification):#

    def __init__(self,output_dim,dropout=0.25):

        super(Longformer, self).__init__()


    
        self.bert_layer = LongformerModel.from_pretrained("yikuan8/Clinical-Longformer",
                                                        gradient_checkpointing=False,
                                                        attention_window = 512 ,output_attentions=True,output_hidden_states=True)
        
        # self.bert_layer = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalGPT-base-zh")
        self.embedding_dim: int = 768
        # self.output_dim: int = 768,
        self.output_dim=output_dim
        self.hidden_dim: int = 256
        self.last_n_layers = 1

        # self.global_embed = GlobalEmbedding(
        #     self.embedding_dim)#, self.hidden_dim, self.output_dim)
       

       
       
        # self.bert_layer.classifier.out_proj=nn.Linear(768, 768) #256
        
        for param in self.bert_layer.parameters():
            
            param.requires_grad = False
        # for param in self.bert_layer.encoder.layer[11].parameters():#score #pooler
        #     param.requires_grad = True
        # for param in self.bert_layer.encoder.layer[10].parameters():#score #pooler
        #     param.requires_grad = True
        
        

        self.dropout1 = nn.Dropout(dropout)
        
        # self.fc1=nn.Linear(1024, 512)
        # self.fc3=nn.Linear(512,256)
        
        # self.fc2=nn.Linear(256, 1)
        # self.linear_layer = nn.Linear(949, 768)



        

    def forward(self, input_ids, attention_mask = None):
        
        # print("****",input_ids.shape)
        
        bert_l=self.bert_layer(input_ids= input_ids, attention_mask=attention_mask)#,return_dict=True,output_attentions=True)
        
        
        
        all_feat = bert_l.last_hidden_state
        
        
        
        report_feat=all_feat[:,0,:]
        
        
        # print("featttt",all_feat.shape,report_feat.shape)
        return  report_feat



class MLPBlock2(nn.Module):
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

class Projector(nn.Module):
    def __init__(self, in_dim,out_dim=128):
        super(Projector, self).__init__()
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
        self.test_layer = nn.Linear(1,12)
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




#self, numerical_columns, categorical_columns,
                #  num_transformer_blocks, num_heads, emb_dim,
                #  attn_dropout_rates, ff_dropout_rates,
                #  mlp_dropout_rates,
                #  mlp_hidden_units_factors,

class Tabtransformer_combined(nn.Module):
    def __init__(self, emb_dim,encoder_categories,categorical_columns,num_transformer_blocks, num_heads, attn_dropout_rates, ff_dropout_rates,mlp_dropout_rates,mlp_hidden_units_factors):
                #  attn_dropout_rates, ff_dropout_rates,
                #  mlp_dropout_rates,
                #  mlp_hidden_units_factors,):
        super().__init__()
        # self.numerical_columns = numerical_columns
        # print("nummm",numerical_columns)
        
        # self.categorical_columns = categorical_columns
        # if "outcome_critical" in self.numerical_columns:
        #     self.numerical_columns.remove("outcome_critical")
        # elif "outcome_critical" in self.categorical_columns:
        #     self.categorical_columns.remove("outcome_critical")
        # self.encoder_categories = encoder_categories
        self.emb_dim = emb_dim
        self.encoder_categories = encoder_categories

        self.embed_layers = nn.ModuleDict()
        
        for i, categorical in enumerate(categorical_columns):
            embedding = nn.Embedding(
                num_embeddings=len(self.encoder_categories[i]),
                embedding_dim=self.emb_dim,
            )
            
            self.embed_layers[categorical] = embedding
        
    def forward(self,df, numerical_columns, categorical_columns):
     
        # self.embed_layers = nn.ModuleDict()
        
        # for i, categorical in enumerate(categorical_columns):
        #     embedding = nn.Embedding(
        #         num_embeddings=len(self.encoder_categories[i]),
        #         embedding_dim=self.emb_dim,
        #     )
            
        #     self.embed_layers[categorical] = embedding
        x_nums = []
        for numerical in numerical_columns:
            x_num = torch.unsqueeze(df[numerical], dim=1)
            x_nums.append(x_num)
        if len(x_nums) > 0:
            x_nums = torch.cat(x_nums, dim=1)
        else:
            x_nums = torch.tensor(x_nums, dtype=torch.float32)
        
        x_cats = []
        for categorical in categorical_columns:
            
            x_cat = self.embed_layers[categorical](df[categorical])
            x_cats.append(x_cat)
        if len(x_cats) > 0:
            x_cats = torch.cat(x_cats, dim=1)
        else:
            x_cats = torch.tensor(x_cats, dtype=torch.float32)
      
        return x_nums, x_cats
 


class report_ehr_model_combined(nn.Module): 
    def __init__(self, num_embeddings,
                 num_transformer_blocks, num_heads, emb_dim,
                 attn_dropout_rates, ff_dropout_rates,
                 mlp_dropout_rates,
                 mlp_hidden_units_factors,num_cat_cols, num_num_cols
                 ):
        super().__init__()
        self.num_num_cols = num_num_cols
        self.num_cat_cols = num_cat_cols
        self.report_transformer = Longformer(output_dim=64)
        self.transformers = nn.Sequential()
       
        self.emb_dim = emb_dim
        

        self.embed_layers = nn.ModuleDict()
        
        for i in range(len(num_embeddings)):
            embedding = nn.Embedding(
                num_embeddings=num_embeddings[i],
                embedding_dim=self.emb_dim,
            )
            
            self.embed_layers[str(i)] = embedding
        for i in range(num_transformer_blocks):
            self.transformers.add_module(f'transformer_{i}', 
                                        TabTransformerBlock(num_heads,
                                                            emb_dim,
                                                            attn_dropout_rates[i],
                                                            ff_dropout_rates[i]))
        
        self.flatten = nn.Flatten()
        self.num_norm = nn.LayerNorm(num_num_cols)
        
        self.n_features = (num_cat_cols * emb_dim) + num_num_cols
        mlp_hidden_units = [int(factor * self.n_features) \
                            for factor in mlp_hidden_units_factors]
        self.mlp = MLPBlock(self.n_features, mlp_hidden_units,
                            mlp_dropout_rates)
        self.test_layer = nn.Linear(1,12)
        self.final_sigmoid = nn.Sigmoid() 
        self.report_projector = Projector(768)
        self.ehr_projector = Projector(128)
        
    
    def forward(self, input_ids,attention_mask,df):
        
        x_nums = []
        df = df.squeeze(1)
        
        for numerical in range(self.num_num_cols):
            x_num = torch.unsqueeze(df[:,numerical], dim=1)
            x_nums.append(x_num)
        if len(x_nums) > 0:
            x_nums = torch.cat(x_nums, dim=1)
        else:
            x_nums = torch.tensor(x_nums, dtype=torch.float32)
        
        x_cats = []
        for categorical in range(self.num_num_cols,self.num_num_cols + self.num_cat_cols):#(97,107):
            
            x_cat = self.embed_layers[str(categorical-self.num_num_cols)](torch.unsqueeze(df[:,categorical],dim=1).long())
            x_cats.append(x_cat)
        if len(x_cats) > 0:
            x_cats = torch.cat(x_cats, dim=1)
        else: # if we have enough number of categorical variables
            x_cats = torch.tensor(x_cats, dtype=torch.float32)
        
        contextualized_x_cats = self.transformers(x_cats)
        contextualized_x_cats = self.flatten(contextualized_x_cats)
        
        if x_nums.shape[-1] > 0:
            x_nums = self.num_norm(x_nums) 
            # x_nums = x_nums.unsqueeze(-1)
            # x_nums = self.test_layer(x_nums)
            # print("cat_num",x_nums.shape,contextualized_x_cats.shape)
            # x_nums = self.flatten(x_nums)
            features = torch.cat((x_nums, contextualized_x_cats), -1)

        else: #### remove this part
            features = contextualized_x_cats
            
        ehr_embedding = self.mlp(features)
       

        report_embedding = self.report_transformer(input_ids,attention_mask)
        report_embedding = self.report_projector(report_embedding)
        ehr_embedding = self.ehr_projector(ehr_embedding)
        # print("embeddings",report_embedding.shape,ehr_embedding.shape)
        return report_embedding,ehr_embedding 

    


