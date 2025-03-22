# Coxnam Pytorch

### Setup Instructions

To set up the virtual environment and install the necessary dependencies, follow these steps:

1. **Set up the virtual environment using `uv`**:
    ```bash
    uv
    ```

2. **Add `scikit-survival` to the environment**:
    ```bash
    uv add scikit-survival
    ```

### Overview

This repository contains the implementation of the CoxNAM model using PyTorch. The CoxNAM model is used for survival analysis and interpretable machine learning on tabular data.

### Usage


1. Clone the project
    ```
    git clone https://github.com/your-username/your-project.git
    cd your-project

    ```

2. Create new virtual env using uv

    ```
    uv env create
    ```

3. Tell UV to use Python 3.10.x

    ```
    uv use python=3.10

    ```

4. Install dependencies listed in `pyproject.toml`:

    ```
    uv install
    
    ```

5. Finally add `uv pip install scikit-survival` to fix dependency conflicts.


### Dataset used

1. SUPPORT2:

This dataset comprises 9105 individual critically ill patients across 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. Each row concerns hospitalized patient records who met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. The goal is to determine these patients' 2- and 6-month survival rates based on several physiologic, demographics, and disease severity information. It is an important problem because it addresses the growing national concern over patients' loss of control near the end of life. It enables earlier decisions and planning to reduce the frequency of a mechanical, painful, and prolonged dying process.

[Download Link](https://datasets.aim-ahead.net/dataset/p/UCI_DS_880)

2. Framingham
The Framingham Heart Study (FHS) is an ongoing cohort study dedicated to identifying common factors or characteristics that contribute to cardiovascular disease (CVD). It is a collaborative project of the National Heart, Lung and Blood Institute and Boston University.

In 1948, an original cohort of 5,209 men and women between 30 and 62 years old were recruited from Framingham, MA. An Offspring Cohort began in 1971, an Omni Cohort in 1994, a Third Generation Cohort in 2002, a New Offspring Spouse Cohort in 2004 and a Second Generation Omni Cohort in 2003. Core research in the dataset focuses on cardiovascular and cerebrovascular diseases. The data include biological specimens, molecular genetic data, phenotype data, samples, images, participant vascular functioning data, physiological data, demographic data, and ECG data.

[Download Link](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)


3. Rossi et al.'s Criminal Recidivism Data
This data set is originally from Rossi et al. (1980), and is used as an example in Allison (1995). The data pertain to 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive aid.


### References

For more information on the CoxNAM model and its applications, refer to the research paper:


Liangchen Xu, Chonghui Guo,
CoxNAM: An interpretable deep survival analysis model,
Expert Systems with Applications,
Volume 227,
2023,
120218,
ISSN 0957-4174

[DOI](https://doi.org/10.1016/j.eswa.2023.120218)

[Link to paper](https://www.sciencedirect.com/science/article/pii/S0957417423007200)

Keywords: Survival analysis; Neural additive model; Interpretable AI; Cox proportion hazards model
