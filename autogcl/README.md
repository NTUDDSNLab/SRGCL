# SAFGCL (AutoGCL)

## Usage
To run the unsupervised learning script, navigate to the `autogcl/unsupervised` directory and execute the following command:
```bash
cd autogcl/unsupervised
CUDA_VISIBLE_DEVICES=0 python us_main_topk.py --dataset='MUTAG'
```

## Parameters
You can adjust the following parameters when running the script:

- `CUDA_VISIBLE_DEVICES`: Set the GPU device to use (e.g., `0` or `1`).
- `--d`: Type of data selector. Default is `'l2_norm'`.
- `--v`: Number of views each generation. Default is `50`.
- `--k`: Top `k` views for contrastive learning. Default is `2`.

Example usage:
```bash
CUDA_VISIBLE_DEVICES=0 python us_main_topk.py --dataset='MUTAG' --d='l2_norm' --v=50 --k=2
```