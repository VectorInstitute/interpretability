
import os
import re
from glob import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
from sklearn.preprocessing import LabelEncoder
import cv2 
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import Sampler

# https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch/tree/master
# https://github.com/cxr-eye-gaze/eye-gaze-dataset
def split_dataset(file_path, random_state=1):
    df = pd.read_csv(file_path)
    # -- Split after uniquing the patient ids so that it does not get split across the different test, dev, test
    pid = list(df['patient_id'].unique())
    random.seed(random_state)
    random.shuffle(pid)
    train_patient_count = round(len(pid) * 0.8)
    not_train = len(pid) - train_patient_count
    # --- Split this remaining equally into dev and test.
    dev_patient_count = round(not_train * 0.5)
    train = df[df['patient_id'].isin(pid[:train_patient_count])]
    dev = df[df['patient_id'].isin(pid[train_patient_count:train_patient_count+dev_patient_count])]
    test = df[df['patient_id'].isin(pid[train_patient_count+dev_patient_count:])]
    return train, dev, test

def read_image(image_path):
    image = cv2.imread(image_path)
    image = image/np.max(image)
    return image



class XrayDataset(Dataset):
    def __init__(self, csv_file, image_path_name):
        
        self.path_name = image_path_name
        self.csv_file = csv_file
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()     
        self.csv_file["numeric_targets"] = self.csv_file['Finding Labels'].apply(lambda x: self.get_tagets(x))   
    def __len__(self):
        
        return len(self.csv_file)
    def choose_the_indices(self):
        
        max_examples_per_class = 10000 # its the maximum number of examples that would be sampled in the training set for any class
        the_chosen = []
        all_classes = {}
        length = len(self.csv_file)
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length),length, replace = False))):
            
            temp = str.split(self.csv_file.iloc[i, :]['Finding Labels'], '|')

            # special case of ultra minority hernia. we will use all the images with 'Hernia' tagged in them.
            if 'Hernia' in temp:
                the_chosen.append(i)
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        all_classes[t] += 1
                continue

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False]*len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    if t in all_classes:
                        if all_classes[t]< max_examples_per_class: # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):                    
                    the_chosen.append(i)
                    # maintain count
                    for t in temp:
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:        # these are single label images
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class: # 500
                            all_classes[t] += 1
                            the_chosen.append(i)

        with open('all_classes.pkl', 'wb') as file:
            pickle.dump(all_classes, file)
        return the_chosen, sorted(list(all_classes)), all_classes
    
    def get_tagets(self,row):
        labels = str.split(row, '|')
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1 
        return target
    def get_image(self, idx):
        # -- Query the index location of the required file
    
        image_name = self.csv_file.loc[idx,'Image Index']
        
        image_path = glob(os.path.join(self.path_name, '**', image_name), recursive=True)[0]
        image = read_image(image_path)
        if len(image.shape) == 2: image = np.expand_dims(image, axis=-1)
        
        labels = str.split(self.csv_file.loc[idx,'Finding Labels'], '|')
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1         
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224
        rseed = 42
        seq = iaa.Sequential([iaa.Resize((input_size, input_size))])
        image_transform = transforms.Compose([seq.augment_image, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])       
        image = image_transform(image)
        return image.float(), target # for all experiments except concept generation
        # return image_path,target   # for concept generation
        

    def num_sort(self, filename):
        not_num = re.compile("\D")
        return int(not_num.sub("", filename))

    def __getitem__(self, idx):
        image, label = self.get_image(idx)
        return image,label # for all experiments except concept generation
        # return image, label,list(self.all_classes)  # for concept generation

# https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/tree/master
class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    This version supports multi-label datasets, where each sample may belong to multiple classes.
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object.
        Args:
        - labels: binary matrix (n_samples x n_classes), where each row represents the labels of a sample.
        - classes_per_it: number of random classes for each iteration.
        - num_samples: number of samples for each iteration for each class (support + query).
        - iterations: number of iterations (episodes) per epoch.
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels  # Binary matrix of size (n_samples x n_classes)
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        # Determine the number of classes and create mappings
        self.num_classes = 15
        self.classes = [torch.tensor(i) for i in range(self.num_classes)]
        # Create a dictionary that maps each class to the indices of samples belonging to it
        self.class_to_indices = {c.item(): [] for c in self.classes} 
        for sample_idx, label_vec in enumerate(self.labels):
        
            for c in torch.nonzero(label_vec).squeeze(1):  # Get active classes for the sample
                self.class_to_indices[c.item()].append(sample_idx)

        # Convert lists to tensors for efficient indexing
        
        for c in self.class_to_indices:
            self.class_to_indices[c] = torch.tensor(self.class_to_indices[c])

    def __iter__(self):
        '''
        Yield a batch of indices.
        '''
        spc = self.sample_per_class  # Samples per class
        cpi = self.classes_per_it    # Classes per iteration

        for _ in range(self.iterations):
            batch_indices = []
            # Randomly sample `cpi` classes
            sampled_classes = torch.randperm(self.num_classes)[:cpi]

            for c in sampled_classes:
                class_indices = self.class_to_indices[c.item()]
                if len(class_indices) >= spc:
                    # Randomly select `spc` samples from this class
                    sampled_indices = class_indices[torch.randperm(len(class_indices))[:spc]]
                else:
                    # Handle rare classes with fewer samples
                    sampled_indices = class_indices
                batch_indices.extend(sampled_indices.tolist())

            # Shuffle the batch indices to ensure randomness
            batch_indices = torch.tensor(batch_indices)
            batch_indices = batch_indices[torch.randperm(len(batch_indices))]
            for idx in batch_indices:
                yield idx.item()
           

    def __len__(self):
        '''
        Return the number of iterations (episodes) per epoch.
        '''
        return self.iterations
