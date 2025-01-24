
import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader,SubsetRandomSampler, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import pickle
import stat
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from model import NeuralAdditiveModel, calculate_n_units, ExULayer, ReLULayer, TabNet, NAM_2, TabTransformer, Preprocessor
# from new_data import dfDataset, process_csv, create_test_train_fold, create_test_train_fold_embed, split_training_dataset, process_csv_transformer, TabDataset
# from losses import penalized_cross_entropy
##
from sklearn.preprocessing import OrdinalEncoder
from model.model import report_ehr_model,Preprocessor, report_ehr_model_combined
from sklearn.model_selection import train_test_split
from dataset.multimodal_dataset import multimodaldataset,process_csv_transformer
from utils.losses import CLIPLoss, cosine_similarity, precision_at_k
from pytorch_tabnet.pretraining import TabNetPretrainer

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch SickKids Brain MRI')

    
    parser.add_argument('--data_path', type=str, default="df.csv", help='Data path')
    parser.add_argument('--output_dir', type=str, default='output_weight', help='Output directory')
    
    parser.add_argument('--batch_size', type=int, default=8, help='batch size') 
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs') #0.0003
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate') #5e-2  #1e-3 #compare:5e-6 #6e-6 (best till now) 0.0002
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler') ##true?
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    parser.add_argument('--feature_dropout', type=int, default=0, help='scheduler step size')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout') 
    parser.add_argument('--n_basis_functions', type=float, default=1000) # 1000
    parser.add_argument('--units_multiplier', type=float, default=32) #2
    parser.add_argument('--hidden_units', type=list, default=[]) 
    parser.add_argument('--output_regularization', type=float, default=0.0)
    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=0.0001) #0.05
    parser.add_argument('--shallow_layer', type=str, default="relu")
    parser.add_argument('--hidden_layer', type=str , default = "relu")
    return parser

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) 
    random.seed(seed_value) 
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(42, True)
splits=KFold(n_splits=5,shuffle=True,random_state=42)

def save_checkpoint(model, optimizer, epoch, loss,file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(model, optimizer,file_path):
    checkpoint = torch.load(file_path)
    # print("checkpointttttttt",checkpoint['model_state_dict'].keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

#each time save a checkpoint
# save a log file  location of the checkpoints
def train_global_local_model(data_ds,numerical_column, categorical_column, num_embeddings):
    
   
    num_batch_accumulate = 8#128#4  
    model_config = {
    'cat_embedding_dim': 12,
    'num_transformer_blocks': 4,#4,
    'num_heads': 3,#3,
    'tf_dropout_rates': [0., 0., 0., 0.,],
    'ff_dropout_rates': [0., 0., 0., 0.,],
    'mlp_dropout_rates': [0.2, 0.1], # might want to change this
    'mlp_hidden_units_factors': [2, 1],
    }
    emb_dim = model_config['cat_embedding_dim']
    num_transformer_blocks = model_config['num_transformer_blocks']
    num_heads = model_config['num_heads']
    attn_dropout_rates = model_config['tf_dropout_rates']
    ff_dropout_rates = model_config['ff_dropout_rates']
    mlp_dropout_rates = model_config['mlp_dropout_rates']
    mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

    global f1_list,auc_list, precision_list, recall_list
    batch_size=args.batch_size
    total_train_auc={}
    total_val_auc={}
    test_auc=[]
    
    # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
    #    print(train_idx,val_idx)
        print("fold:", fold)
        if fold>0: 
            continue
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_dl = DataLoader(data_ds, batch_size=args.batch_size)##, sampler=train_sampler,drop_last=True)
        valid_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=test_sampler,drop_last=True)
    
    
    
        num_embeddings = torch.tensor(num_embeddings).to(0)
        # preprocessor = Preprocessor(emb_dim,encoder_categories,categorical_column)
        num_cat_cols, num_num_cols = len(categorical_column), len(numerical_column)
        model2 = report_ehr_model_combined(
                 num_embeddings,num_transformer_blocks, num_heads, emb_dim,
                 attn_dropout_rates, ff_dropout_rates,
                 mlp_dropout_rates,
                 mlp_hidden_units_factors,num_cat_cols, num_num_cols
                 )
        
        # for i in range(torch.cuda.device_count()):
        #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        #     print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        #     print(f"  Total Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
        #     print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
        model = nn.DataParallel(model2, device_ids= [0,1])
        # model = model.cuda()
        model = model.to(0)
        ita_list=[]
        ita_list_val=[]
        local_list=[]
        cl_list=[]
        # params = list(model.parameters())#list(preprocessor.parameters()) + 
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        # print("checkpointttttttt",preprocessor.state_dict().keys())


        try:
            model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer,"output_weight/new_model_checkpoint_corrected.pth")
            # preprocessor, _, _, _ = load_checkpoint(preprocessor, optimizer,"output_weight/preprocessor_checkpoint.pth")
            print(f"Resuming from epoch {start_epoch + 1}")
        except FileNotFoundError:
            print("No checkpoint found, starting frsom scratch")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        optimizer.zero_grad()
        cl_list=[]
        for epoch in range(args.num_epochs):
            # preprocessor.train()
            model.train()
            train_loss = 0
            counter = 0
            num_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]
            train_loss = 0
            counter = 0
            num_batches=0
            num_batches_valid=0
            val_batches=0
            val_batch=0
            train_batches=0
            epoch_loss = 0
            training_ture=[]
            training_estimated=[]

            
            report_embeddings = []
            ehr_embeddings=[]
            
            i2t_corr_tr=0
            batch_epoch_tr=0
            i2t_corr_val=0
            t2i_corr_tr=0
            t2i_corr_val=0
            batch_epoch_val=0
            precision_list=[]
            loc_list = []
           
            print("epoch",epoch)
            for num_batches,(text,ehr,num_embeddings) in enumerate(train_dl): 
                
                # features ,labels= features.cuda(),labels.cuda()
                optimizer.zero_grad()
                # x_nums, x_cats = preprocessor(ehr, numerical_column, categorical_column)
                # # print("leennnnnnnnnnnnnn",len(x_nums))
                # # print("feature shape",features.shape)

                # # x_nums, x_cats= x_nums.cuda(), x_cats.cuda()
                # # model = model.cuda()
                # x_nums , x_cats= x_nums.to(0) , x_cats.to(0)
                
                # # images=images.float()
                # # logits, fnns_out = model(features)
                
                
                attention_mask = text['attention_mask'].cuda()
                input_ids = text['input_ids'].cuda().squeeze(1)
                attention_mask = attention_mask.to(0)
                input_ids = input_ids.to(0)
                ehr = ehr.to(0)
                # print("ehrrrrr",ehr.shape)
                report_emb,ehr_emb = model(input_ids,attention_mask,ehr)#x_nums.squeeze(), x_cats)
                
                report_embeddings.append(report_emb)
            
                ehr_embeddings.append(ehr_emb)
                

                if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    
                    

                    report_embeddings = torch.cat(report_embeddings, dim=0)
                    ehr_embeddings=torch.cat(ehr_embeddings, dim=0) 
                               
                    bz = len(report_embeddings)
                    
                    # labs = torch.arange(bz).type_as(report_emb).long()
                    # labels = torch.eye(bz).type_as(report_emb)[labs]
                    
                
                    
                    loss_g = CLIPLoss(temperature = 0.1) # triplet loss
                    # print("new embeddings",ehr_embeddings.shape,report_embeddings.shape)
                    loss_global,_,_= loss_g(ehr_embeddings,report_embeddings)#
                    
                    loss0 = loss_global
                    i_t_scores=cosine_similarity(ehr_embeddings,report_embeddings)
                    # print("ittt_scores",i_t_scores.shape)
                    i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
                    # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
                    # print("pr",i2t_acc1_tr) #REMOVE
                    i2t_corr_tr+=i2t_corr_tr_batch
                    
                    
                    batch_epoch_tr+=i2t_batch_tr

                    loss0.backward()


                    optimizer.step() 
                    
                    if (num_batches+1)%8==0:
                        save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/new_model_checkpoint_corrected_backup.pth")
                        # print("saved!")
                    save_checkpoint(model, optimizer, epoch, loss0.item(), file_path=f"output_weight/new_model_checkpoint_corrected.pth")
                    # save_checkpoint(preprocessor, optimizer, epoch, loss0.item(), file_path="output_weight/preprocessor_checkpoint.pth")
                    optimizer.zero_grad() 
                    train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
                    train_batches+=1

                    
                    ehr_embeddings = []
                    report_embeddings = []
                    
                    
                
                counter += 1

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
    ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
            # scheduler.step()
            train_loss = train_loss/(train_batches)#*batch_size)  
            print("loss_ita",train_loss)
            ita_list.append(train_loss)
            # scheduler.step()
            # global only: _if_updated_location_attention_local_global__fold
            if epoch%5==0 and epoch>9:
                torch.save(model.state_dict(), os.path.join(args.output_dir,f"_ssl_model_{fold}__epoch__{epoch}"))
                # torch.save(preprocessor.state_dict(), os.path.join(args.output_dir,f"_ssl_preprocessor_{fold}__epoch__{epoch}__margin{args.margin}"))
            del loss0 ### Sajith 
            i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
            # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
            print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
            precision_list.append(i2t_precision_tr)
            model.eval()
#             with torch.set_grad_enabled(False):

#                 val_loss = 0.0
#                 val_b=0
#                 total_epoch = 0
#                 validation_true = []
#                 validation_estimated = []
#                 n = 0
#                 # valid_dl.dataset.dataset.test_or_val = True
#                 image_embeddings_val = []
#                 report_embeddings_val = []
    
#                 word_attention_val=[]
#                 word_embeddings_val=[]
#                 patch_embeddings_val=[]
#                 sent_list_val=[]
#                 patch_attention_list_val=[]
#                 cap_lens_list_val=[]
#                 patch_weights_val=[]
#                 word_weights_val=[]
#                 merged_att_list_val=[]
#                 val_loc_list = []
#                 for num_batches_valid,(images, text,labells,masks,tumor_location) in enumerate(valid_dl):
                    
#  #               for i, batch in enumerate(valid_dl):
#      #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
#       #              model.to(device)
#             #_ = model(batch[0][0])
#     #                if torch.cuda.is_available():
#                     images = images.to(device)
#                     mask = text['attention_mask'].to(device)
#                     input_id = text['input_ids'].squeeze(1).to(device)
                    
#                     # pred,img_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)
#                     img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)

#                     image_embeddings_val.append(img_emb_q)
#                     report_embeddings_val.append(report_emb_q)
#                     word_attention_val.append(word_atten_output)
#                     word_embeddings_val.append(word_emb_q)
#                     patch_embeddings_val.append(patch_emb_q)
#                     sent_list_val.append(sents)
#                     patch_attention_list_val.append(patch_atten_output)
#                     cap_lens_list_val.append(cap_lens)
#                     patch_weights_val.append(patch_atten_output_weight)
#                     word_weights_val.append(word_atten_output_weight)
#                     merged_att_list_val.append(merged_att)
#                     val_loc_list.append(tumor_location)
#                     # bz = img_emb_q.size(0)
                   
#                     # prob = torch.sigmoid(pred)

#                     # labs = torch.arange(bz).type_as(report_emb_q).long()
#                     # labels = torch.eye(bz).type_as(report_emb_q)[labs]
#                     # loss_fn = ContrastiveLoss_euclidean(margin=0.1)
#                     # loss0 = loss_fn(img_emb_q,report_emb_q, labels)
#                     # valid_loss = loss0
                    
                    
#                     # for i in range(len(labells.tolist())):
#                     #     validation_true.append(labells.tolist()[i])#[0])
#                     #     validation_estimated.append(prob.tolist()[i])#[0])

#                     if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): 
                        
#                         # val_batches = 0  
#                         cap_len_list_val = [element for sublist in cap_lens_list_val for element in sublist]
#                         val_batches+=1
#                         image_embeddings_val = torch.cat(image_embeddings_val, dim=0)
#                         report_embeddings_val = torch.cat(report_embeddings_val, dim=0)
#                         word_attention_val=torch.cat(word_attention_val, dim=0)
#                         word_embeddings_val=torch.cat(word_embeddings_val, dim=0)
#                         patch_embeddings_val=torch.cat(patch_embeddings_val, dim=0)
#                         # sent_list_val=torch.cat(sent_list_val, dim=0)
#                         patch_attention_list_val=torch.cat(patch_attention_list_val, dim=0)
#                         patch_weights_val=torch.cat(patch_weights_val, dim=0)
#                         # sent_list_val=torch.cat(sent_list_val, dim=0)
#                         word_weights_val=torch.cat(word_weights_val, dim=0)
#                         merged_att_list_val=torch.cat(merged_att_list_val,dim=0)
#                         val_loc_list=torch.cat(val_loc_list,dim=0)
#                         bz = len(image_embeddings_val)
                   
#                         # prob = torch.sigmoid(pred)

#                         labs = torch.arange(bz).type_as(report_embeddings_val).long()
#                         labels = torch.eye(bz).type_as(report_embeddings_val)[labs]

#                         # euclidean_distance = torch.nn.functional.pairwise_distance(image_embeddings_val, report_embeddings_val, keepdim=True)
                        
#                         # loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
#                         if args.similarity_measure=="euclidian":
#                             valid_loss_g = ContrastiveLoss_euclidean(margin=args.margin)
#                         else:
#                             valid_loss_g = ContrastiveLoss_cosine2(margin=args.margin,mode="global")
                        
                        

#                         valid_loss_global,i_t_scores,t_i_scores = valid_loss_g(image_embeddings_val,report_embeddings_val, labels,val_loc_list)
#                         # i2t_acc1_val = precision_at_k(i_t_scores, labels, top_k=(1,))
#                         # t2i_acc1_val = precision_at_k(t_i_scores, labels, top_k=(1,))
#                         # print("validation precision",(i2t_acc1_val + t2i_acc1_val) / 2.)

#                         valid_loss_local_0 = local_contrastive_loss2(word_embeddings_val, patch_embeddings_val,patch_attention_list_val, cap_len_list_val,val_loc_list,margin=args.margin)
#                         valid_loss_local_1 = local_contrastive_loss2(patch_embeddings_val, word_embeddings_val,word_attention_val, cap_len_list_val,val_loc_list,margin=args.margin)
#                         # valid_loss_1 = loss_fn(report_embeddings_val,image_embeddings_val, labels)
#                         # valid_loss = (valid_loss_0 + valid_loss_1)/2
#                         # valid_loss = loss_fn(euclidean_distance, labels)
#                         # valid_loss = loss0
#                         valid_loss_local=valid_loss_local_0+valid_loss_local_1
#                         valid_losses = torch.cat([valid_loss_global.view(1), valid_loss_local.view(1)])

#                         # weighted_losses = valid_losses * loss_weights.unsqueeze(0).cuda()
#                         # valid_loss = torch.sum(weighted_losses)
#                         valid_loss = valid_loss_local + valid_loss_global
#                         i2t_acc1_val,i2t_corr_val_batch,i2t_batch_val = precision_at_k(i_t_scores)#, labels, top_k=(1,))
#                         # t2i_acc1_val,t2i_corr_val_batch,t2i_batch_val = precision_at_k(t_i_scores)#, labels, top_k=(1,))
#                         i2t_corr_val+=i2t_corr_val_batch
#                         batch_epoch_val+=i2t_batch_val
#                         # t2i_corr_val+=t2i_corr_val_batch

#                         val_loss += valid_loss.detach().item()
#                         image_embeddings_val=[]
#                         report_embeddings_val=[]
#                         word_attention_val=[]
#                         word_embeddings_val=[]
#                         patch_embeddings_val=[]
#                         sent_list_val=[]
#                         patch_attention_list_val=[]
#                         cap_lens_list_val=[]
#                         patch_weights_val=[]
#                         word_weights_val=[]
#                         merged_att_list_val=[]
#                         val_loc_list = []
#     #                corr = (pred > 0.0).squeeze().long() != labels
#                     # val_err += int(corr.sum())
#     #                total_epoch += len(labels)
#                     n = n + 1
#                 val_loss = val_loss / (val_batches)#* batch_size)
#                 print("validation_ita",val_loss)

#                 i2t_precision_val=i2t_corr_val/batch_epoch_val
#                 # t2i_precision_val=t2i_corr_val/batch_epoch_val
#                 print("validation precision", (i2t_precision_val))#+t2i_precision_val)/2)
                
#                 ita_list_val.append(val_loss)
#                 test_true = []
#                 test_estimated = []
#                 # for images, labells in test_dl:
                        
                    
#                 #     images, labells = images.to(device), labells.to(device)
#                 #     pred = model(images)
#                 #     prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
#                 #     for i in range(len(labells.tolist())):
#                 #         test_true.append(labells.tolist()[i][0])
#                 #         test_estimated.append(prob.tolist()[i][0])

#                 # Calculate the AUC for the different models
#     #        print("sssssssssssssssssssssssss",validation_true)#len(validation_true))
#             # val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
#             # # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
#             # train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

#                 # total_val_err[epoch] = float(val_err) / total_epoch
#     #            total_val_loss[epoch] = float(val_loss) / (n + 1)


        

            
#             ##### self-added
            
#             if epoch >=50 and epoch%10==0:
#                 print("ita_train_list",ita_list)
#                 print("ita_valid_list",ita_list_val)
#                 print("precision_list",precision_list)
#             del valid_loss
#         model.eval()
#         with torch.set_grad_enabled(False):

#             # val_loss = 0.0
#             # val_b=0
#             # total_epoch = 0
#             test_true = []
#             test_estimated = []
#             n = 0
#             # test_dl.dataset.dataset.test_or_val = True
            
#             for images, text,labells,masks,tumor_location in test_dl:
                
# #               for i, batch in enumerate(valid_dl):
#     #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
#     #              model.to(device)
#         #_ = model(batch[0][0])
# #                if torch.cuda.is_available():
#                 images= images.to(device)
#                 mask = text['attention_mask'].to(device)
#                 input_id = text['input_ids'].squeeze(1).to(device)
        
#                 img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)#(**texts)#
               
#                 # pred = model(images)
#                 # def model_wrapper(*x):
#                 #     return model(*x)[0]
#                 # print(model) 
#                 # print("layer",model.layer4[0].conv2)               
#                 # layer_gc = LayerGradCam(model, model.layer4[0].conv2) #layer4[0].conv1
#                 # # layer_gc.grad_cam.forward_func = model_wrapper
#                 # # layer_gc.guided_backprop.forward_func = model_wrapper
#                 # attr = layer_gc.attribute(images)
#                 # print(model)
#                 # print("attr",attr.shape)
                
#                 ##$$$$upsampled_attr = LayerAttribution.interpolate(attr, (240,240,155))
#                 # print("attr",upsampled_attr.shape)
#                 ####$$$$numpy_heatmap=upsampled_attr .cpu().detach().numpy()
#                 # print(numpy_heatmap[0][0].shape)
#                 # for saving:
#                 # for i in numpy_heatmap:
#                 ###$$$$ np.save(os.path.join(args.output_dir,"model_heatmap.npy"),numpy_heatmap[0])#[i][0])
        
                
#                 #F.softmax(pred, dim=1)
#                 # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                
                
            
#         if fold==0:
#             break

#     #        train_fpr, train_tpr, _ = roc_curve(training_true, training_estimated)
#     #        val_fpr, val_tpr, _ = roc_curve(validation_true, validation_estimated)

            

#     #        logging.info(
#     #            "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} ".format(
#     #                epoch + 1,
#     #                total_train_loss[epoch],
#     #                total_train_auc[epoch],
#     #                total_val_loss[epoch],
#     #                total_val_auc[epoch]))

#             # model_path = get_model_name(trial=fold, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
#                                         # dropout_rate=0.1, epoch=epoch + 1) #after mednet commented

#         # if fold==3:
#         # if 
                

#         # torch.save(model.state_dict(), os.path.join(args.output_dir,f"_cl_attention__fold__{fold}"))

#     # np.savetxt(os.path.join(save_folder, "{}_train_err.csv".format(model_path)), total_train_err)
# #    np.savetxt(os.path.join(args.output_dir, "{}_train_loss.csv".format(model_path)), total_train_loss)
# #    np.savetxt(os.path.join(args.output_dir, "{}_train_auc.csv".format(model_path)), total_train_auc)
#     # np.savetxt(os.path.join(save_folder, "{}_val_err.csv".format(model_path)), total_val_err)
# #    np.savetxt(os.path.join(args.output_dir, "{}_val_loss.csv".format(model_path)), total_val_loss)
# #    np.savetxt(os.path.join(args.output_dir, "{}_val_auc.csv".format(model_path)), total_val_auc)



#     logging.info('Finished training.')
# #    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

#     # return total_train_err, total_train_loss, total_train_auc, total_val_err, total_val_loss, total_val_auc
    return 0

# def train_global_local_model(data_ds,numerical_column, categorical_column, encoder_categories):
    
   
#     num_batch_accumulate = 8#128#4  
#     model_config = {
#     'cat_embedding_dim': 12,
#     'num_transformer_blocks': 4,#4,
#     'num_heads': 3,#3,
#     'tf_dropout_rates': [0., 0., 0., 0.,],
#     'ff_dropout_rates': [0., 0., 0., 0.,],
#     'mlp_dropout_rates': [0.2, 0.1],
#     'mlp_hidden_units_factors': [2, 1],
#     }
#     emb_dim = model_config['cat_embedding_dim']
#     num_transformer_blocks = model_config['num_transformer_blocks']
#     num_heads = model_config['num_heads']
#     attn_dropout_rates = model_config['tf_dropout_rates']
#     ff_dropout_rates = model_config['ff_dropout_rates']
#     mlp_dropout_rates = model_config['mlp_dropout_rates']
#     mlp_hidden_units_factors = model_config['mlp_hidden_units_factors']

#     global f1_list,auc_list, precision_list, recall_list
#     batch_size=args.batch_size
#     total_train_auc={}
#     total_val_auc={}
#     test_auc=[]
    
#     # test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
#     for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data_ds)))):
#     #    print(train_idx,val_idx)
#         print("fold:", fold)
#         if fold>0: 
#             continue
#         print('Fold {}'.format(fold + 1))
#         train_sampler = SubsetRandomSampler(train_idx)
#         test_sampler = SubsetRandomSampler(val_idx)
#         train_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=train_sampler,drop_last=True)
#         valid_dl = DataLoader(data_ds, batch_size=args.batch_size, sampler=test_sampler,drop_last=True)
    
    
    

#         preprocessor = Preprocessor(emb_dim,encoder_categories,categorical_column)
#         model2 = report_ehr_model(numerical_column, categorical_column,
#                  num_transformer_blocks, num_heads, emb_dim,
#                  attn_dropout_rates, ff_dropout_rates,
#                  mlp_dropout_rates,
#                  mlp_hidden_units_factors,
#                  )
        
#         # for i in range(torch.cuda.device_count()):
#         #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#         #     print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
#         #     print(f"  Total Memory (GB): {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
#         #     print(f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}")
#         model = nn.DataParallel(model2, device_ids= [0,1])
#         # model = model.cuda()
#         model = model.to(0)
#         ita_list=[]
#         ita_list_val=[]
#         local_list=[]
#         cl_list=[]
#         params = list(preprocessor.parameters()) + list(model.parameters())
#         optimizer = optim.AdamW(params, lr=args.lr,weight_decay=args.weight_decay)
#         # print("checkpointttttttt",preprocessor.state_dict().keys())
#         try:
#             model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer,"output_weight/model_checkpoint.pth")
#             preprocessor, _, _, _ = load_checkpoint(preprocessor, optimizer,"output_weight/preprocessor_checkpoint.pth")
#             print(f"Resuming from epoch {start_epoch + 1}")
#         except FileNotFoundError:
#             print("No checkpoint found, starting frsom scratch")

#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
#         optimizer.zero_grad()
#         cl_list=[]
#         for epoch in range(args.num_epochs):
#             preprocessor.train()
#             model.train()
#             train_loss = 0
#             counter = 0
#             num_batches=0
#             val_batch=0
#             train_batches=0
#             epoch_loss = 0
#             training_ture=[]
#             training_estimated=[]
#             train_loss = 0
#             counter = 0
#             num_batches=0
#             num_batches_valid=0
#             val_batches=0
#             val_batch=0
#             train_batches=0
#             epoch_loss = 0
#             training_ture=[]
#             training_estimated=[]

            
#             report_embeddings = []
#             ehr_embeddings=[]
            
#             i2t_corr_tr=0
#             batch_epoch_tr=0
#             i2t_corr_val=0
#             t2i_corr_tr=0
#             t2i_corr_val=0
#             batch_epoch_val=0
#             precision_list=[]
#             loc_list = []
           
#             print("epoch",epoch)
#             for num_batches,(text,ehr) in enumerate(train_dl): 
        
#                 # features ,labels= features.cuda(),labels.cuda()
                
#                 x_nums, x_cats = preprocessor(ehr, numerical_column, categorical_column)
#                 # print("leennnnnnnnnnnnnn",len(x_nums))
#                 # print("feature shape",features.shape)

#                 # x_nums, x_cats= x_nums.cuda(), x_cats.cuda()
#                 # model = model.cuda()
#                 x_nums , x_cats= x_nums.to(0) , x_cats.to(0)
#                 optimizer.zero_grad()
#                 # images=images.float()
#                 # logits, fnns_out = model(features)
               
#                 attention_mask = text['attention_mask'].cuda()
#                 input_ids = text['input_ids'].cuda().squeeze(1)
#                 attention_mask = attention_mask.to(0)
#                 input_ids = input_ids.to(0)
#                 report_emb,ehr_emb = model(input_ids,attention_mask,x_nums.squeeze(), x_cats)
                
#                 report_embeddings.append(report_emb)
            
#                 ehr_embeddings.append(ehr_emb)
                

#                 if ((num_batches + 1) % num_batch_accumulate == 0) or (num_batches + 1 == len(train_dl)):
                    
                    

#                     report_embeddings = torch.cat(report_embeddings, dim=0)
#                     ehr_embeddings=torch.cat(ehr_embeddings, dim=0) 
                               
#                     bz = len(report_embeddings)
                    
#                     # labs = torch.arange(bz).type_as(report_emb).long()
#                     # labels = torch.eye(bz).type_as(report_emb)[labs]
                    
                
                    
#                     loss_g = CLIPLoss(temperature = 0.1)
#                     # print("new embeddings",ehr_embeddings.shape,report_embeddings.shape)
#                     loss_global,_,_= loss_g(ehr_embeddings,report_embeddings)#
                    
#                     loss0 = loss_global
#                     i_t_scores=cosine_similarity(ehr_embeddings,report_embeddings)
#                     # print("ittt_scores",i_t_scores.shape)
#                     i2t_acc1_tr,i2t_corr_tr_batch,i2t_batch_tr = precision_at_k(i_t_scores)#, labels, top_k=(1,))
#                     # t2i_acc1_tr,t2i_corr_tr_batch,t2i_batch_tr  = precision_at_k(t_i_scores)#, labels, top_k=(1,))
#                     print("pr",i2t_acc1_tr)
#                     i2t_corr_tr+=i2t_corr_tr_batch
                    
                    
#                     batch_epoch_tr+=i2t_batch_tr

#                     loss0.backward()


#                     optimizer.step() 
                    
#                     save_checkpoint(model, optimizer, epoch, loss0.item(), file_path="output_weight/model_checkpoint.pth")
#                     save_checkpoint(preprocessor, optimizer, epoch, loss0.item(), file_path="output_weight/preprocessor_checkpoint.pth")
#                     optimizer.zero_grad() 
#                     train_loss+=loss0.detach().item()#*batch_size ### Sajith: added detach 
#                     train_batches+=1

                    
#                     ehr_embeddings = []
#                     report_embeddings = []
                    
                    
                
#                 counter += 1

#             # Calculate average over epoch
#             # total_train_err[epoch] = float(train_err) / total_epoch
#     ##        total_train_loss[epoch] = float(train_loss) / (n + 1)
#             # scheduler.step()
#             train_loss = train_loss/(train_batches)#*batch_size)  
#             print("loss_ita",train_loss)
#             ita_list.append(train_loss)
#             # scheduler.step()
#             # global only: _if_updated_location_attention_local_global__fold
#             if epoch%10==0 and epoch>100:
#                 torch.save(model.state_dict(), os.path.join(args.output_dir,f"_ssl_model_{fold}__epoch__{epoch}"))
#                 torch.save(preprocessor.state_dict(), os.path.join(args.output_dir,f"_ssl_preprocessor_{fold}__epoch__{epoch}__margin{args.margin}"))
#             del loss0 ### Sajith 
#             i2t_precision_tr=i2t_corr_tr/batch_epoch_tr
#             # t2i_precision_tr=t2i_corr_tr/batch_epoch_tr
#             print("training precision", (i2t_precision_tr))#+t2i_precision_tr)/2)
#             precision_list.append(i2t_precision_tr)
#             model.eval()
# #             with torch.set_grad_enabled(False):

# #                 val_loss = 0.0
# #                 val_b=0
# #                 total_epoch = 0
# #                 validation_true = []
# #                 validation_estimated = []
# #                 n = 0
# #                 # valid_dl.dataset.dataset.test_or_val = True
# #                 image_embeddings_val = []
# #                 report_embeddings_val = []
    
# #                 word_attention_val=[]
# #                 word_embeddings_val=[]
# #                 patch_embeddings_val=[]
# #                 sent_list_val=[]
# #                 patch_attention_list_val=[]
# #                 cap_lens_list_val=[]
# #                 patch_weights_val=[]
# #                 word_weights_val=[]
# #                 merged_att_list_val=[]
# #                 val_loc_list = []
# #                 for num_batches_valid,(images, text,labells,masks,tumor_location) in enumerate(valid_dl):
                    
# #  #               for i, batch in enumerate(valid_dl):
# #      #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
# #       #              model.to(device)
# #             #_ = model(batch[0][0])
# #     #                if torch.cuda.is_available():
# #                     images = images.to(device)
# #                     mask = text['attention_mask'].to(device)
# #                     input_id = text['input_ids'].squeeze(1).to(device)
                    
# #                     # pred,img_emb_q,report_emb_q,patch_emb_q,word_feat_q,word_attn_q,sents,patch_atten_output,word_atten_output = model(images.float(),input_id, mask)
# #                     img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)

# #                     image_embeddings_val.append(img_emb_q)
# #                     report_embeddings_val.append(report_emb_q)
# #                     word_attention_val.append(word_atten_output)
# #                     word_embeddings_val.append(word_emb_q)
# #                     patch_embeddings_val.append(patch_emb_q)
# #                     sent_list_val.append(sents)
# #                     patch_attention_list_val.append(patch_atten_output)
# #                     cap_lens_list_val.append(cap_lens)
# #                     patch_weights_val.append(patch_atten_output_weight)
# #                     word_weights_val.append(word_atten_output_weight)
# #                     merged_att_list_val.append(merged_att)
# #                     val_loc_list.append(tumor_location)
# #                     # bz = img_emb_q.size(0)
                   
# #                     # prob = torch.sigmoid(pred)

# #                     # labs = torch.arange(bz).type_as(report_emb_q).long()
# #                     # labels = torch.eye(bz).type_as(report_emb_q)[labs]
# #                     # loss_fn = ContrastiveLoss_euclidean(margin=0.1)
# #                     # loss0 = loss_fn(img_emb_q,report_emb_q, labels)
# #                     # valid_loss = loss0
                    
                    
# #                     # for i in range(len(labells.tolist())):
# #                     #     validation_true.append(labells.tolist()[i])#[0])
# #                     #     validation_estimated.append(prob.tolist()[i])#[0])

# #                     if ((num_batches_valid + 1) % num_batch_accumulate == 0) or (num_batches_valid + 1 == len(valid_dl)): 
                        
# #                         # val_batches = 0  
# #                         cap_len_list_val = [element for sublist in cap_lens_list_val for element in sublist]
# #                         val_batches+=1
# #                         image_embeddings_val = torch.cat(image_embeddings_val, dim=0)
# #                         report_embeddings_val = torch.cat(report_embeddings_val, dim=0)
# #                         word_attention_val=torch.cat(word_attention_val, dim=0)
# #                         word_embeddings_val=torch.cat(word_embeddings_val, dim=0)
# #                         patch_embeddings_val=torch.cat(patch_embeddings_val, dim=0)
# #                         # sent_list_val=torch.cat(sent_list_val, dim=0)
# #                         patch_attention_list_val=torch.cat(patch_attention_list_val, dim=0)
# #                         patch_weights_val=torch.cat(patch_weights_val, dim=0)
# #                         # sent_list_val=torch.cat(sent_list_val, dim=0)
# #                         word_weights_val=torch.cat(word_weights_val, dim=0)
# #                         merged_att_list_val=torch.cat(merged_att_list_val,dim=0)
# #                         val_loc_list=torch.cat(val_loc_list,dim=0)
# #                         bz = len(image_embeddings_val)
                   
# #                         # prob = torch.sigmoid(pred)

# #                         labs = torch.arange(bz).type_as(report_embeddings_val).long()
# #                         labels = torch.eye(bz).type_as(report_embeddings_val)[labs]

# #                         # euclidean_distance = torch.nn.functional.pairwise_distance(image_embeddings_val, report_embeddings_val, keepdim=True)
                        
# #                         # loss_fn = ContrastiveLoss_euclidean(margin=args.margin)
# #                         if args.similarity_measure=="euclidian":
# #                             valid_loss_g = ContrastiveLoss_euclidean(margin=args.margin)
# #                         else:
# #                             valid_loss_g = ContrastiveLoss_cosine2(margin=args.margin,mode="global")
                        
                        

# #                         valid_loss_global,i_t_scores,t_i_scores = valid_loss_g(image_embeddings_val,report_embeddings_val, labels,val_loc_list)
# #                         # i2t_acc1_val = precision_at_k(i_t_scores, labels, top_k=(1,))
# #                         # t2i_acc1_val = precision_at_k(t_i_scores, labels, top_k=(1,))
# #                         # print("validation precision",(i2t_acc1_val + t2i_acc1_val) / 2.)

# #                         valid_loss_local_0 = local_contrastive_loss2(word_embeddings_val, patch_embeddings_val,patch_attention_list_val, cap_len_list_val,val_loc_list,margin=args.margin)
# #                         valid_loss_local_1 = local_contrastive_loss2(patch_embeddings_val, word_embeddings_val,word_attention_val, cap_len_list_val,val_loc_list,margin=args.margin)
# #                         # valid_loss_1 = loss_fn(report_embeddings_val,image_embeddings_val, labels)
# #                         # valid_loss = (valid_loss_0 + valid_loss_1)/2
# #                         # valid_loss = loss_fn(euclidean_distance, labels)
# #                         # valid_loss = loss0
# #                         valid_loss_local=valid_loss_local_0+valid_loss_local_1
# #                         valid_losses = torch.cat([valid_loss_global.view(1), valid_loss_local.view(1)])

# #                         # weighted_losses = valid_losses * loss_weights.unsqueeze(0).cuda()
# #                         # valid_loss = torch.sum(weighted_losses)
# #                         valid_loss = valid_loss_local + valid_loss_global
# #                         i2t_acc1_val,i2t_corr_val_batch,i2t_batch_val = precision_at_k(i_t_scores)#, labels, top_k=(1,))
# #                         # t2i_acc1_val,t2i_corr_val_batch,t2i_batch_val = precision_at_k(t_i_scores)#, labels, top_k=(1,))
# #                         i2t_corr_val+=i2t_corr_val_batch
# #                         batch_epoch_val+=i2t_batch_val
# #                         # t2i_corr_val+=t2i_corr_val_batch

# #                         val_loss += valid_loss.detach().item()
# #                         image_embeddings_val=[]
# #                         report_embeddings_val=[]
# #                         word_attention_val=[]
# #                         word_embeddings_val=[]
# #                         patch_embeddings_val=[]
# #                         sent_list_val=[]
# #                         patch_attention_list_val=[]
# #                         cap_lens_list_val=[]
# #                         patch_weights_val=[]
# #                         word_weights_val=[]
# #                         merged_att_list_val=[]
# #                         val_loc_list = []
# #     #                corr = (pred > 0.0).squeeze().long() != labels
# #                     # val_err += int(corr.sum())
# #     #                total_epoch += len(labels)
# #                     n = n + 1
# #                 val_loss = val_loss / (val_batches)#* batch_size)
# #                 print("validation_ita",val_loss)

# #                 i2t_precision_val=i2t_corr_val/batch_epoch_val
# #                 # t2i_precision_val=t2i_corr_val/batch_epoch_val
# #                 print("validation precision", (i2t_precision_val))#+t2i_precision_val)/2)
                
# #                 ita_list_val.append(val_loss)
# #                 test_true = []
# #                 test_estimated = []
# #                 # for images, labells in test_dl:
                        
                    
# #                 #     images, labells = images.to(device), labells.to(device)
# #                 #     pred = model(images)
# #                 #     prob = torch.sigmoid(pred)#F.softmax(pred, dim=1)
# #                 #     for i in range(len(labells.tolist())):
# #                 #         test_true.append(labells.tolist()[i][0])
# #                 #         test_estimated.append(prob.tolist()[i][0])

# #                 # Calculate the AUC for the different models
# #     #        print("sssssssssssssssssssssssss",validation_true)#len(validation_true))
# #             # val_auc = roc_auc_score(validation_true, validation_estimated)#,multi_class='ovr')
# #             # # test_auc = roc_auc_score(test_true, test_estimated)#,multi_class='ovr')
# #             # train_auc = roc_auc_score(training_ture, training_estimated)#,multi_class='ovr')

# #                 # total_val_err[epoch] = float(val_err) / total_epoch
# #     #            total_val_loss[epoch] = float(val_loss) / (n + 1)


        

            
# #             ##### self-added
            
# #             if epoch >=50 and epoch%10==0:
# #                 print("ita_train_list",ita_list)
# #                 print("ita_valid_list",ita_list_val)
# #                 print("precision_list",precision_list)
# #             del valid_loss
# #         model.eval()
# #         with torch.set_grad_enabled(False):

# #             # val_loss = 0.0
# #             # val_b=0
# #             # total_epoch = 0
# #             test_true = []
# #             test_estimated = []
# #             n = 0
# #             # test_dl.dataset.dataset.test_or_val = True
            
# #             for images, text,labells,masks,tumor_location in test_dl:
                
# # #               for i, batch in enumerate(valid_dl):
# #     #               model = medcam.inject(model, output_dir=gradcam_dir, save_maps=True)
# #     #              model.to(device)
# #         #_ = model(batch[0][0])
# # #                if torch.cuda.is_available():
# #                 images= images.to(device)
# #                 mask = text['attention_mask'].to(device)
# #                 input_id = text['input_ids'].squeeze(1).to(device)
        
# #                 img_emb_q,report_emb_q,patch_emb_q,word_emb_q,sents,patch_atten_output_weight,word_atten_output_weight,cap_lens,patch_atten_output,word_atten_output,merged_att = model(images.float(),input_id, mask)#(**texts)#
               
# #                 # pred = model(images)
# #                 # def model_wrapper(*x):
# #                 #     return model(*x)[0]
# #                 # print(model) 
# #                 # print("layer",model.layer4[0].conv2)               
# #                 # layer_gc = LayerGradCam(model, model.layer4[0].conv2) #layer4[0].conv1
# #                 # # layer_gc.grad_cam.forward_func = model_wrapper
# #                 # # layer_gc.guided_backprop.forward_func = model_wrapper
# #                 # attr = layer_gc.attribute(images)
# #                 # print(model)
# #                 # print("attr",attr.shape)
                
# #                 ##$$$$upsampled_attr = LayerAttribution.interpolate(attr, (240,240,155))
# #                 # print("attr",upsampled_attr.shape)
# #                 ####$$$$numpy_heatmap=upsampled_attr .cpu().detach().numpy()
# #                 # print(numpy_heatmap[0][0].shape)
# #                 # for saving:
# #                 # for i in numpy_heatmap:
# #                 ###$$$$ np.save(os.path.join(args.output_dir,"model_heatmap.npy"),numpy_heatmap[0])#[i][0])
        
                
# #                 #F.softmax(pred, dim=1)
# #                 # loss = classifier_criterion(pred, labells.unsqueeze(1).float()) 
                
                
                
            
# #         if fold==0:
# #             break

# #     #        train_fpr, train_tpr, _ = roc_curve(training_true, training_estimated)
# #     #        val_fpr, val_tpr, _ = roc_curve(validation_true, validation_estimated)

            

# #     #        logging.info(
# #     #            "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} ".format(
# #     #                epoch + 1,
# #     #                total_train_loss[epoch],
# #     #                total_train_auc[epoch],
# #     #                total_val_loss[epoch],
# #     #                total_val_auc[epoch]))

# #             # model_path = get_model_name(trial=fold, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
# #                                         # dropout_rate=0.1, epoch=epoch + 1) #after mednet commented

# #         # if fold==3:
# #         # if 
                

# #         # torch.save(model.state_dict(), os.path.join(args.output_dir,f"_cl_attention__fold__{fold}"))

# #     # np.savetxt(os.path.join(save_folder, "{}_train_err.csv".format(model_path)), total_train_err)
# # #    np.savetxt(os.path.join(args.output_dir, "{}_train_loss.csv".format(model_path)), total_train_loss)
# # #    np.savetxt(os.path.join(args.output_dir, "{}_train_auc.csv".format(model_path)), total_train_auc)
# #     # np.savetxt(os.path.join(save_folder, "{}_val_err.csv".format(model_path)), total_val_err)
# # #    np.savetxt(os.path.join(args.output_dir, "{}_val_loss.csv".format(model_path)), total_val_loss)
# # #    np.savetxt(os.path.join(args.output_dir, "{}_val_auc.csv".format(model_path)), total_val_auc)



# #     logging.info('Finished training.')
# # #    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

# #     # return total_train_err, total_train_loss, total_train_auc, total_val_err, total_val_loss, total_val_auc
#     return 0


if __name__ == "__main__":

    args = make_parser().parse_args()
    auc_list , precision_list, recall_list, f1_list = [], [], [], []
    df = pd.read_csv("../ssl_df.csv")
    # print(df.columns)
    # print(salam)
    tech = 0
    print("shappppeeppe",df.shape)
    cap_list = []
    # for text in df["text"]:
        
    #     cap_list.extend(find_all_caps_words(text))
    # print("setttttt",set(cap_list))
    df =df.dropna(axis = 1)
    df['ed_los'] = pd.to_timedelta(df['ed_los']).dt.seconds / 60
    df = df.drop(columns="ed_los")
    df,df_num_cols, df_cat_cols  = process_csv_transformer(df)
    # bool_columns = df.select_dtypes(include='bool')
    # df[df_cat_cols_non_bools] = df[df_cat_cols].fillna("unknown")
    # df[df_num_cols] = df[df_num_cols].fillna(-1)

    # print("null columns",df.iloc[:,[6, 9, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26]].isna().sum())
    # for col in df.columns:
    #     print(col,df[col].isna().sum())

    # df = df.drop(df.columns[19],axis=1)
    # df = df.drop(df.columns[22],axis=1)
    
    
    # for col in [19,22]:#[6, 9, 12, 15, 19, 21, 22, 23, 24, 25, 26, 27, 28]:
    #     print(df.iloc[:,col].isna().sum())
    # df_cat_cols.remove("discharge_location")
    # df_cat_cols.remove("ethnicity")
    # print("null",df.isna().sum())

    df_cat_cols.remove("outcome_ed_revisit_3d")
    df_cat_cols.remove("outcome_hospitalization")
    text = df["text"]
    df = df.drop(columns=["text","outcome_ed_revisit_3d","outcome_hospitalization"])
    print("kkkkkkkkkkkk",df_cat_cols)
    oe = OrdinalEncoder(handle_unknown='error',
                dtype=np.int64)
    df_cat_cols.remove("text")
    
    encoded = oe.fit_transform(df[df_cat_cols].values)
    with open('ordinal_encoder.pkl', 'wb') as file:
        pickle.dump(oe, file)
    df[df_cat_cols] = encoded
    
    encoder_categories = oe.categories_
    num_embedding_list = []
    for i, categorical in enumerate(df_cat_cols):

            num_embedding_list.append(len(encoder_categories[i])),
    
    # df[df_cat_cols] = oe.fit_transform(df[df_cat_cols].values)
    
    df.index=range(df.shape[0])
    
    print("nummmm",df_num_cols)
    ds = multimodaldataset(df,df_num_cols,df_cat_cols,num_embedding_list,text)

    with open('num_emb.pkl', 'wb') as file:
        pickle.dump(num_embedding_list, file)
    
    
    _ = train_global_local_model(ds,df_num_cols, df_cat_cols,num_embedding_list)


# if __name__ == "__main__":

#     args = make_parser().parse_args()
#     auc_list , precision_list, recall_list, f1_list = [], [], [], []
#     df = pd.read_csv("../ssl_df.csv")
#     tech = 0
#     print("shappppeeppe",df.shape)
#     cap_list = []
#     # for text in df["text"]:
        
#     #     cap_list.extend(find_all_caps_words(text))
#     # print("setttttt",set(cap_list))
#     df =df.dropna(axis = 1)
     
    
#     df,df_num_cols, df_cat_cols  = process_csv_transformer(df)
#     # bool_columns = df.select_dtypes(include='bool')
#     # df[df_cat_cols_non_bools] = df[df_cat_cols].fillna("unknown")
#     # df[df_num_cols] = df[df_num_cols].fillna(-1)

#     # print("null columns",df.iloc[:,[6, 9, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26]].isna().sum())
#     # for col in df.columns:
#     #     print(col,df[col].isna().sum())

#     # df = df.drop(df.columns[19],axis=1)
#     # df = df.drop(df.columns[22],axis=1)
    
    
#     # for col in [19,22]:#[6, 9, 12, 15, 19, 21, 22, 23, 24, 25, 26, 27, 28]:
#     #     print(df.iloc[:,col].isna().sum())
#     # df_cat_cols.remove("discharge_location")
#     # df_cat_cols.remove("ethnicity")
#     # print("null",df.isna().sum())
#     oe = OrdinalEncoder(handle_unknown='error',
#                 dtype=np.int64)
#     df_cat_cols.remove("text")
#     encoded = oe.fit_transform(df[df_cat_cols].values)
#     df[df_cat_cols] = encoded
#     encoder_categories = oe.categories_
    
#     df[df_cat_cols] = oe.fit_transform(df[df_cat_cols].values)
    
#     df.index=range(df.shape[0])
    

#     ds = multimodaldataset(df,df_num_cols,df_cat_cols)


#     _ = train_global_local_model(ds,df_num_cols, df_cat_cols, encoder_categories)
