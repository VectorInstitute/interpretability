# Some parts were extracted from https://github.com/ChantalMP/Xplainer?tab=readme-ov-file
from dataset.class_prompts import class_prompts
from model.zeroshot import InferenceModel
import argparse
import gc
from pathlib import Path
from utils.utils import get_roc_auc_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from dataset.dataset import EyegazeDataset
from pathlib import Path
import pickle
from utils.zeroshot_utils import calculate_auroc

def make_parser():
    parser = argparse.ArgumentParser(description='Zero-shot Classification (concept-based explainability)')

    # Data
    # parser.add_argument('--data_path', type=str, default='resources/master_sheet.csv', help='Data path')
    # parser.add_argument('--image_path', type=str, default='/data/MIMIC/MIMIC-IV/cxr_v2/physionet.org/files/mimic-cxr/2.0.0', help='image_path')
    
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['Normal', 'CHF', 'pneumonia'], help='Label names for classification')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')

   

    # Misc
    parser.add_argument('--gpus', type=str, default='7', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--gcam_viz', default=False, action='store_true', help='[USE] Used for displaying the GradCam results')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--weight_decay', type=int, default=1e-3, help='Seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=20, help='Seed for reproducibility')
    return parser

def zeroshot(test_ds):

    dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)#, collate_fn=lambda x: x, num_workers=0)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(class_prompts)

    all_labels = []
    all_probs_neg = []

    for images, labels, keys in tqdm(dataloader):
        
        agg_probs = []
        agg_negative_probs = []
        
        image_paths = [Path(image_path) for image_path in images]
        for image_path in image_paths:
            
            probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
            agg_probs.append(probs)
            agg_negative_probs.append(negative_probs)

        probs = {}  # Aggregated
        negative_probs = {}  # Aggregated
        for key in agg_probs[0].keys():
            probs[key] = sum([p[key] for p in agg_probs]) / len(agg_probs)  # Mean Aggregation

        for key in agg_negative_probs[0].keys():
            negative_probs[key] = sum([p[key] for p in agg_negative_probs]) / len(agg_negative_probs)  # Mean Aggregation

        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(class_prompts, pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(class_prompts,
                                                                                                   disease_probs=disease_probs,
                                                                                                   negative_disease_probs=negative_disease_probs,
                                                                                                   keys= agg_probs[0].keys())
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        
    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    # evaluation
    all_labels = all_labels.squeeze(1)
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    

    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    
    print(f"AUROC: {overall_auroc:.5f}\n")
    




if __name__ == '__main__':
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chexpert', help='chexpert or chestxray14')
    args = parser.parse_args()

    image_path = "/datasets/nih-chest-xrays"    
    csv_file = pd.read_csv(os.path.join(image_path,"Data_Entry_2017.csv"))
    test_split = os.path.join(image_path,"test_list.txt")
    with open(test_split, 'r') as f:
        test_images = f.read().splitlines()
    test_df = csv_file[csv_file['Image Index'].isin(test_images)]
    test_df.reset_index(drop=True, inplace=True)
    test_ds = EyegazeDataset(test_df, image_path)
    zeroshot(test_ds)


   

