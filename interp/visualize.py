import matplotlib.pyplot as plt
import torch.nn.functional as F


def visualize_trainable_attention(attention_weights, image, num_classes):

    fig, ax = plt.subplots()
    attn_map_upscaled = F.interpolate(
                            attention_weights.mean(0).view(7,7).unsqueeze(0).unsqueeze(0).clamp(0, 1),  # Add batch and channel dims
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False)

    ax.add_image(image.permute(1, 2, 0).cpu().clamp(0, 1))  # Original image
    ax.add_image(attn_map_upscaled.squeeze(0).permute(1, 2, 0).cpu(), cmap='jet', alpha=0.5)  # Attention overlay
    for ind in range(num_classes):
        #if pred[ind]==1:
        plt.savefig(f"trainable_attention.png_{ind}.png")
