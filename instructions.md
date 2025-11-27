This is a step-by-step guide to run the examples and training code present in the repository. Any specific instructions to run a code within a folder will be present in the folder's `README.md`.

## Running on the cluster

1. Login to the cluster via `ssh` and 2FA
```bash
ssh username@v.vectorinstitute.ai
```

2. Clone the repository in your `home` directory

```bash
git clone git@github.com:VectorInstitute/interpretability-bootcamp.git
```

3. Create a Jupyter kernel to use common bootcamp environment present on the cluster.
```bash
source  /ssd003/projects/aieng/public/interp_bootcamp/venv/bin/activate
ipython kernel install --user --name=interp_bootcamp
deactivate
```

**[Optional]:** You can create your own environment using the instructions below and create a kernel with it.

### Use Jupyter Hub for Notebooks

4. Connect to Vector's VPN and login to the jupyter hub:
https://vdm1.cluster.local:8000/.

5. Open any notebook from the cloned repo, change kernel to `interp_bootcamp` and run it.

### Use commandline to run python scripts
6. Run the following command on the cluster to login to a GPU node.
```bash
srun --pty --mem=30GB -c 5 --gres=gpu:1 --qos=normal -t 8:00:00 /bin/bash
```
7. Source the bootcamp virtual environment.
```bash
source  /ssd003/projects/aieng/public/interp_bootcamp/venv/bin/activate
```

**[Optional]:** You can also source your own environment if you have created it manually.

8. Run any python script within the repository. For e.g.
```python
python ~/interpretability-bootcamp/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_explain.py
```

## Running on your laptop

1. Clone the repository on your system

```bash
git clone git@github.com:VectorInstitute/interpretability-bootcamp.git
```
2. Create your own virtual environment.

via `pip`
- [Optional]: Install Python 3.10 if its not installed.
- Run the following commands

```bash
python3 -m venv interp_bootcamp
source interp_bootcamp/bin/activate
cd interpretability-bootcamp
pip install -r requirements.txt
```

via `uv`

- Install `uv` on your MAC or cluster. [Link](https://docs.astral.sh/uv/getting-started/installation/)
- [Optional] Install python 3.10 via uv. [Link](https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version)
- Create the environment
```bash
cd interpretability-bootcamp
uv venv --python 3.10
uv sync
```
- Source the environment to run the files
```bash
source .venv/bin/activate
```

3. Run any python script within the repository. For e.g.
```python
python ~/interpretability-bootcamp/reference_implementations/Intepretable-models/Imaging/B-Cos/isic_explain.py
```

4. Create a jupyter kernel

if env created via `pip`
```bash
source interp_bootcamp/bin/activate
ipython kernel install --user --name=interp_bootcamp
deactivate
```

if env created via `uv`
```bash
source .venv/bin/activate
ipython kernel install --user --name=interp_bootcamp
deactivate
```

5. Run jupyter server.

- Source environment

```bash
source interp_bootcamp/bin/activate
```

or

```bash
source .venv/bin/activate
```

- Start server
```bash
jupyter notebook
```

6. Change kernel in the jupyter notebook to `interp_bootcamp` and run it.
