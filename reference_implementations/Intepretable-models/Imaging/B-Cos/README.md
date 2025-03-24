## Method
- This is an demonstration of a simple Resnet-50 B-Cos network for interpretable ML. In this example, a ImageNet pre-trained ResNet50 B-Cos network is finetuned on the ISIC-2016 Dermascopy dataset for malignant/benign skin cancer classification. 

## Resources
- [isic_finetune.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_fine_tune.py) to load the pretrained resnet(bcos) network and fine-tune it.
- [isic_explanations](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_explain.py) to visualize the predictions image by the fine-tuned network.
- [isic_explanations-1.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_explanations-1.png), [isic_explanations.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_explanations.png) Image with original image and predicted heatmap.
- [bcos.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/B-Cos/bcos.yaml) yaml file with configs to run bcos training scripts.
- [bcos](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Imaging/B-Cos/bcos) folder contains class files for the dataset, bcos model class etc.
- [models](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Imaging/B-Cos/models) contains pretrained and finetuned weights for BCOS model.

## Datasets
- The dataset can be downloaded from the [ISIC 2016 Challenge website](https://challenge.isic-archive.com/data/). 
- Specifically, we shall use the dataset and ground truths for Task 3 - Skin Lesion Classification Challenge.

1. [Training dataset](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip): 900 dermoscopic lesion images in JPEG format.

2. [Training Ground Truth Labels](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv): 900 entries of gold standard malignant status labels.

3. [Test dataset](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip): 379 images of the exact same format as the Training Data.

4. [Test Set Ground Truth Labels](https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv): 379 entries of gold standard malignant status labels.


- If you use this dataset, please cite:

Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian; Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin Imaging Collaboration (ISIC)". eprint arXiv:1605.01397. 2016.

 
