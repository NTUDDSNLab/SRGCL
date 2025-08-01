# SRGCL(GraphCL)
## Parameters
You can adjust the following parameters when running the example script:

- `CUDA_VISIBLE_DEVICES`: Set the GPU device to use (e.g., `0` or `1`).
- `--DS`: Dataset to use. Options are `'MUTAG'`, `'NCI1'`, `'PTC_MR'`, `'IMDB-BINARY'`, `'IMDB-MULTI'`, `'REDDIT-BINARY'`. Default is `'MUTAG'`.
- `--r`: Augmentation ratio. Default is `0.2`.
- `--d`: Type of data selector. Default is `'l2_norm'`.
- `--v`: Number of views each generation. Default is `50`.
- `--k`: Top `k` views for contrastive learning. Default is `2`.
- `--aug`: Type of augmentation. Default is `'dnodes'`.
- `--epochs`: Number of training epochs. Default is `30`.
- `--init_temp`: Set initial temperature. Default is `1.0`.
- `--exp_factor`: Exponential method factor. Default is `0.1`.
- `--decay_type`: Type of learning rate decay. Options are `'exponential'`, `'cosine'`. Default is `'exponential'`.

## Example usage
```bash
CUDA_VISIBLE_DEVICES=0 python gcl_gcl_method6_temperature.py --DS='MUTAG' --d='l2_norm' --v=50 --k=2
```
