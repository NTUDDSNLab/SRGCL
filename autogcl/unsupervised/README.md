# SRGCL(AutoGCL)
## Parameters
You can adjust the following parameters when running the example script:

- `CUDA_VISIBLE_DEVICES`: Set the GPU device to use (e.g., `0` or `1`).
- `--dataset`: Dataset to use. Options are `'MUTAG'`, `'NCI1'`, `'PTC_MR'`, `'IMDB-BINARY'`, `'IMDB-MULTI'`, `'REDDIT-BINARY'`. Default is `'MUTAG'`.
- `--r`: Augmentation ratio. Default is `0.2`.
- `--d`: Type of data selector. Default is `'l2_norm'`.
- `--v`: Number of views each generation. Default is `50`.
- `--k`: Top `k` views for contrastive learning. Default is `2`.
- `--aug`: Type of augmentation. Default is `'dnodes'`.
- `--epochs`: Number of training epochs. Default is `30`.
- `--init_temp`: Set initial temperature. Default is `1.0`.
- `--exp_factor`: Exponential method factor. Default is `0.1`.
- `--decay_type`: Type of learning rate decay. Options are `'exponential'`, `'cosine'`. Default is `'exponential'`.
- `--start_deterministic`: The epoch starts to use exactly topk in temperature sampling. Default is `20`.
- `--cosine_factor`: Cosine method factor. Default is `0.5`.

## Example usage
```bash
CUDA_VISIBLE_DEVICES=0 python us_main_temperature_hybrid2.py --dataset='MUTAG' --d='l2_norm' --v=50 --k=2
```
