# SAFGCL(GraphCL)
## Parameters
You can adjust the following parameters when running the example script:

- `CUDA_VISIBLE_DEVICES`: Set the GPU device to use (e.g., `0` or `1`).
- `--d`: Type of data selector. Default is `'l2_norm'`.
- `--v`: Number of views each generation. Default is `50`.
- `--k`: Top `k` views for contrastive learning. Default is `2`.
- `--aug`: Type of augmentation. Default is `'dnodes'`.
- `--r`: Augmentation ratio. Default is `0.2`.
- `--epochs`: Number of training epochs. Default is `30`.

## Example usage
```bash
CUDA_VISIBLE_DEVICES=0 python gcl_gcl_method6_topk.py --DS='MUTAG' --d='l2_norm' --v=50 --k=2
```