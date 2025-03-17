import os

import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize_trainable_attention(attention,
                                  image,
                                  pred,
                                  num_classes,
                                  save_plots: bool = True,
                                  save_dir: str = 'attention_plots'):
    """
    """
    _, _ = plt.subplots()

    attn_map_upscaled = F.interpolate(
                        attention.mean(0).view(7,7).unsqueeze(0).unsqueeze(0).clamp(0, 1),  # Add batch and channel dims
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False)

    plt.imshow(image.permute(1, 2, 0).cpu().clamp(0, 1))  # Original image
    overlay = attn_map_upscaled.squeeze(0).permute(1, 2, 0).cpu()
    plt.imshow(overlay, cmap='jet', alpha=0.5)  # Attention overlay

    if save_plots:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for ind in range(num_classes):
            if pred[ind]==1:
                plt.savefig(f"{save_dir}/attention_for_class_{ind}.png")
    else:
        plt.show()