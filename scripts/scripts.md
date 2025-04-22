This file contains instructions to submit a `SLURM` job for training models. They allocate GPUs for running scripts and run them. 

## Steps to run job

1. Login to the cluster via `ssh` and 2FA
```bash
ssh username@v.vectorinstitute.ai
```

2. Clone the repository in your `home` directory

```bash
git clone git@github.com:VectorInstitute/interpretability-bootcamp.git
```

3. Change directory to `scripts`.
```bash
cd scripts
```
4. Source environment on the cluster
```bash
source  /ssd003/projects/aieng/public/interp_bootcamp/venv/bin/activate
```

5. Submit training job to SLURM.
```bash
sbatch train_nam.sh
```

## Debugging

1. 2 files will be created for every job submitted:
- `.out` file containing output of the script being run
- `.err` file containing any errors thrown by the script.

You can use the following commands to view the output - 
- `tail -f filename.out`
- `cat filename.err`

2. To view the status of the slurm job:
`squeue --me`


