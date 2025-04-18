import random
from torch.utils.data import Dataset

class MNistDataset(Dataset):

    def __init__(self, tupple, image_path_name):
        super(MNistDataset, self).__init__()
        self.tupple = tupple
        self.image_path_name = image_path_name 
        self.images = self.tupple[0]
        self.targets = self.tupple[1]
    def __len__(self):
        
        return len(self.images)
        
    def get_image(self, idx):
        # -- Query the index location of the required file
    
        image = self.tupple[0][idx]
        target = self.tupple[1][idx]
        return image.float(), target
    def __getitem__(self, idx):
        image, label = self.get_image(idx)
        return image,label

def create_episode(dataset, n_way, k_shot, q_query):
    classes = random.sample(set(dataset.targets.numpy()), n_way)  # Unique classes
    support_set, query_set = [], []
    
    for i, cls in enumerate(classes):
        cls_indices = (dataset.targets == cls).nonzero(as_tuple=True)[0]
        selected_indices = random.sample(cls_indices.tolist(), k_shot + q_query)
        support_set.extend([(dataset[idx][0].unsqueeze(0), i) for idx in selected_indices[:k_shot]])
        query_set.extend([(dataset[idx][0].unsqueeze(0), i) for idx in selected_indices[k_shot:]])    
    return support_set, query_set

def load_mnist_data(image_path)
    file_path_train = os.path.join(image_path,"training.pt")
    x_train =torch.load(file_path_train)
    file_path_test = os.path.join(image_path,"test.pt")
    x_test =torch.load(file_path_test)
    train_ds = MNistDataset(x_train, image_path)
    test_ds = MNistDataset(x_test, image_path)
    return train_ds, test_ds
