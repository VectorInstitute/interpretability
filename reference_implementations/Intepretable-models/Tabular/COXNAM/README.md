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
