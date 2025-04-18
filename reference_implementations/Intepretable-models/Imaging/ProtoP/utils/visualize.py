import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import cv2



def prototype_heatmap(xray_image,image_reps,assigned_proto):

    image_rep = F.normalize(image_reps.float(), p=2, dim=1)  # Normalize along the channels
    assigned_proto = F.normalize(assigned_proto.float(), p=2, dim=1)  # Normalize along the channels
    # Reshape prototype representation to a spatial map (8x8) and normalize
    prototype_map = assigned_proto.reshape(8, 8).cpu().detach().numpy()
    prototype_map = (prototype_map - prototype_map.min()) / (prototype_map.max() - prototype_map.min())

    # Resize prototype map to match the original image size (28x28)
    prototype_resized = cv2.resize(prototype_map, (28, 28), interpolation=cv2.INTER_CUBIC)
    similarity_map = F.cosine_similarity(image_rep, assigned_proto, dim=0)  # Similarity along the channel dimension
    similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())
    resized_similarity_map = torch.nn.functional.interpolate(
    similarity_map.reshape(8,8).unsqueeze(0).unsqueeze(0),  #112
        size=xray_image.size()[-2:],  # Height and Width of the image
        mode='bilinear',
        align_corners=False
        ).squeeze()
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_similarity_map.detach().cpu()), cv2.COLORMAP_VIRIDIS)
    prototype_colored = cv2.applyColorMap(np.uint8(255 * prototype_resized), cv2.COLORMAP_JET)
    # Remove channel dimension if it exists
    if xray_image.ndim == 3 and xray_image.shape[0] == 1:
        xray_image = xray_image.squeeze(0)  # Shape becomes (H, W)
    # Convert grayscale image to 3-channel
    input_image_colored = cv2.cvtColor(np.uint8(255 * xray_image.detach().cpu()), cv2.COLOR_GRAY2BGR)

    # Overlay heatmap and prototype onto the original image

    overlayed_image_heatmap = cv2.addWeighted(input_image_colored, 0.5, heatmap_colored, 0.5, 0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(xray_image.squeeze(0).detach().cpu(), cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(resized_similarity_map.detach().cpu().numpy(), cmap="viridis")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(prototype_resized, cmap="gray")
    plt.title("Prototype Visualization")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(overlayed_image_heatmap, cv2.COLOR_BGR2RGB), cmap="viridis")
    plt.title("Image + Heatmap")
    plt.axis("off")

    plt.savefig("prototype_heatmap.png")
        















    





