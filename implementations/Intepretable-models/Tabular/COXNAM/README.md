## Method

This repository contains the implementation of the CoxNAM model using PyTorch. The CoxNAM model is used for survival analysis and interpretable machine learning on tabular data.

## Resources

- [coxnam](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Tabular/COXNAM/coxnam) folder contains scripts for coxnam model, loss functions, metric functions etc.
- [train](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train) folder contains :
    - [coxnam.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/coxnam.yaml) contains parameters for model training, dataset paths etc.
    - [framingham_shape_functions.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/framingham_shape_functions.png) is an image showing shape plots for framingham dataset.
    - [train.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/train.py) is a script to train CoxNAM model on recividism dataset and plot shape functions.
    - [train_support.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/train_support.py) is a script to train CoxNAM model on support2 dataset and plot shape functions.
    - [train_framingham.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/train_framingham.py) is a script to train CoxNAM model on framingham dataset and plot shape functions.
    - [train_framingham_10year.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/train_framingham_10year.py) is a script to train CoxNAM model on a 10 year framingham dataset

## Running the scripts
- Change file paths and parameters values within the [coxnam.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/COXNAM/train/coxnam.yaml) file.
- Login to a GPU node on the cluster
```bash
srun --pty --mem=20GB -c 2 --gres=gpu:1 --qos=normal -t 8:00:00 /bin/bash
```
- Source the virtual environment
```bash
source /ssd003/projects/aieng/public/interp_bootcamp/venv/bin/activate
```
- Run script via `python` command
```bash
python train_support.py
```
## Dataset

1. SUPPORT2:

This dataset comprises 9105 individual critically ill patients across 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. Each row concerns hospitalized patient records who met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. The goal is to determine these patients' 2- and 6-month survival rates based on several physiologic, demographics, and disease severity information. It is an important problem because it addresses the growing national concern over patients' loss of control near the end of life. It enables earlier decisions and planning to reduce the frequency of a mechanical, painful, and prolonged dying process.

[Download Link](https://datasets.aim-ahead.net/dataset/p/UCI_DS_880)

2. Framingham
The Framingham Heart Study (FHS) is an ongoing cohort study dedicated to identifying common factors or characteristics that contribute to cardiovascular disease (CVD). It is a collaborative project of the National Heart, Lung and Blood Institute and Boston University.

In 1948, an original cohort of 5,209 men and women between 30 and 62 years old were recruited from Framingham, MA. An Offspring Cohort began in 1971, an Omni Cohort in 1994, a Third Generation Cohort in 2002, a New Offspring Spouse Cohort in 2004 and a Second Generation Omni Cohort in 2003. Core research in the dataset focuses on cardiovascular and cerebrovascular diseases. The data include biological specimens, molecular genetic data, phenotype data, samples, images, participant vascular functioning data, physiological data, demographic data, and ECG data.

[Download Link](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

3. Rossi et al.'s Criminal Recidivism Data
This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.