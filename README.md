# SRGCL: Self-Reinforcement Graph Contrastive Learning

## Requirement

```shell
python 3.9.16
torch 2.4.1
torchvision 0.19.1
```

## Project Directory Structure

- `autogcl/`
- `data/` - Directory for storing datasets
- `graphcl/`
- `.gitignore`
- `README.md`

Note: Place your data in the current directory.

## GraphCL

### Run
Please check the README in graphcl/ for more details.

```shell
cd graphcl
CUDA_VISIBLE_DEVICES=0 python gcl_gcl_method6_temperature.py --DS='MUTAG' --aug='dnodes'
```

## AutoGCL

### Run
Please check the README in autogcl/ for more details.

```shell
cd autogcl/unsupervised
CUDA_VISIBLE_DEVICES=0 python us_main_temperature_hybrid2.py --dataset='MUTAG'
```
