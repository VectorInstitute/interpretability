import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms

def visualize_trainable_attention(attention_weights, image,label,pred):

    fig, ax = plt.subplots()
    
    attn_map_upscaled = F.interpolate(
    attention_weights.mean(0).view(7,7).unsqueeze(0).unsqueeze(0).clamp(0, 1),  # Add batch and channel dims
    size=(224, 224),
    mode='bilinear',
    align_corners=False)#.squeeze() 
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    

    ax.imshow(image.permute(1, 2, 0).cpu().clamp(0, 1))  # Original image
    ax.imshow(attn_map_upscaled.squeeze(0).permute(1, 2, 0).cpu(), cmap='jet', alpha=0.5)  # Attention overlay
    for ind in range(15):
        if pred[ind]==1:
            plt.savefig(f"trainable_attention.png_{ind}.png")

def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open('all_classes.pkl', 'rb') as all_classes:
        all_classes = pickle.load(all_classes)
    
    NoFindingIndex = all_classes.get('No Finding', -1)
    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            class_roc_auc_list.append(class_roc_auc)
            if i != NoFindingIndex:
                useful_classes_roc_auc_list.append(class_roc_auc)
    return np.mean(np.array(useful_classes_roc_auc_list))

def prototype_heatmap(xray_image,image_reps,prototype_rep):


    similarities = torch.mm(image_reps, prototype_rep.T) 
    assigned_prototypes = prototype_rep[torch.argmax(similarities, dim=1),:]
    
    for i, (image,image_rep) in enumerate(zip(xray_image,image_reps)):
        assigned_proto = prototype_rep[torch.argmax(similarities[i,:]),:].unsqueeze(0)
        image_rep = F.normalize(image_rep.unsqueeze(0).float(), p=2, dim=1)  # Normalize along the channels
        assigned_proto = F.normalize(assigned_proto.float(), p=2, dim=1)  # Normalize along the channels

        # Calculate cosine similarity at each pixel
        similarity_map = F.cosine_similarity(image_rep, assigned_proto, dim=0)  # Similarity along the channel dimension
    
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())
        resized_similarity_map = torch.nn.functional.interpolate(
            similarity_map.reshape(112,112).unsqueeze(0).unsqueeze(0),
            size=image.size()[-2:],  # Height and Width of the image
            mode='bilinear',
            align_corners=False
            ).squeeze()
        plt.imshow(image[0].cpu().numpy(), cmap='gray')
        plt.imshow(resized_similarity_map.cpu().detach().numpy(), cmap='jet', alpha=0.5)
        plt.colorbar(label='Similarity')
        plt.axis('off')
        plt.show()
        plt.savefig("prototype_heatmap.png")
        break

def counterfactual_explanation(model, image):

    explainer = CounterfactualExplainer(
    model=model
    )

    explanations = explainer.explain(image)
    explanations.ipython_plot(index=0) 


    






