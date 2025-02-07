## Interpretable ML using B-Cos Networks ##


- This is an demonstration of a simple Resnet-50 B-Cos network for interpretable ML. In this example, a ImageNet pre-trained ResNet50 B-Cos network is finetuned on the ISIC-2016 Dermascopy dataset for malignant/benign skin cancer classification. 

- Execute `isic_finetune.py` to load the pretrained resnet(bcos) network and fine-tune it.
- Execute `isic_explanations` to visualize the predictions mage by the fine-tuned network.