import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, attention=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.attention = SelfAttention(out_channels) if attention else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        
        if self.attention:
            out = self.attention(out)

        return out

# Self-Attention block
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        # calculating attention weights
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        # calculating weighted context
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x # weighted context added to the original input in a residual manner
        return out, attention

# ResNet with Self-Attention
class ResNetAttention(nn.Module):
    def __init__(self, original_model,num_classes=15):
        super(ResNetAttention, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.attention = SelfAttention(in_dim=512)  # Adjust channels based on ResNet block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)  # Output channels should match in_channels of attention layer
        self.fc2 = nn.Linear(256, num_classes) 

    def forward(self, x):
        x = self.features(x)
        x, attention = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, attention
    
# extracted from: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/protonet.py 
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    

class text_generation():
    def __init__(self):
        # Load the model and tokenizer
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def generate_caption(self,image):
        # Process the image
        image_array = np.array(image.convert("RGB"))
        
        # image_array = np.clip(image_array, a_min=image_array.min(), a_max=image_array.max())  # Ensure no extreme values
        if len(image_array.shape) == 2:
            # Stack to create 3 channels (RGB)
            image_array = np.stack([image_array] * 3, axis=0)
        normalized_image = image_array
        # normalized_image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # normalized_image = ((image_array + 1.0) * 255/2)#.astype(np.uint8)
        ## visualize the image
        # plt.imshow(image_array[0],cmap="gray")  
        # plt.colorbar()  # Add a colorbar for reference
        # plt.title('Visualization of the NumPy Array')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')

        # Save the visualization
        # plt.savefig('array_visualization.png', dpi=300)
        # self.processor.do_rescale = False
        text = "a chest x-ray showing " 
        inputs = self.processor(normalized_image,text, return_tensors="pt")
        pixel_values = inputs.pixel_values
        # Generate caption
        # generated_ids = self.model.generate(pixel_values=pixel_values, max_length=200, num_beams=6, temperature=1.5,early_stopping=False,repetition_penalty=2.0, do_sample=True)
        # generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        generated_ids = self.model.generate(**inputs,max_length=200, num_beams=8, temperature=0.7,early_stopping=False,repetition_penalty=2.0, do_sample=True,top_k=50)
        generated_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"generated text:{generated_caption}")
        
        return generated_caption
    

