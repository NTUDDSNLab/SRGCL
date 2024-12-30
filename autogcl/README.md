# SAFGCL(AutoGCL)
cd autogcl/unsupervised
CUDA_VISIBLE_DEVICES=0 python us_main_topk.py --dataset='MUTAG'

CUDA_VISIBLE_DEVICES=0 or 1
--d : default='l2_norm', help='Types of data selector'
--v : default=50, help='number of views each generation'
--k : default=2, help='Top k views for contrastive learning'