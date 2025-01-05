import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', default='MUTAG', help='MUTAG,NCI1,PTC_MR,IMDB-BINARY,IMDB-MULTI,REDDIT-BINARY')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    parser.add_argument('--device', default='cuda:0', type=str, help='gpu device ids')
    parser.add_argument('--lr', dest='lr', type=float, default= 0.01,
            help='Learning rate.')
    parser.add_argument('--alpha', default=1.2, type=float, help='stregnth for regularization')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--aug', type=str, default='dnodes', help='dnodes, pedges, mask_nodes, hybrid')
    parser.add_argument('--r', default=0.2, type=float, help='aug_ratio')
    parser.add_argument('--v', type=int, default=50, help='number of views each generation')
    parser.add_argument('--k', type=int, default=2, help='Top k views for contrastive learning')
    parser.add_argument('--d', type=str, default='l2_norm', help='Types of data selector (cosine, l2_norm)')
    parser.add_argument('--eta', type=float, default=1.0, help='0.1, 1.0, 10, 100, 1000')
    parser.add_argument('--batch_size', type=int, default=128, help='128, 256, 512, 1024')
    parser.add_argument('--start_deterministic', type=int, default=20, help='The epoch starts to use exactly topk in temperature sampling')
    parser.add_argument('--decay_type', type=str, default='exponential', help='exponential, cosine')
    parser.add_argument('--log_interval', type=int, default=10)
    return parser.parse_args()


