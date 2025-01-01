import torch
import numpy as np
import matplotlib.pyplot as plt
from arguments import arg_parse


def check_dimensional_collapse(embeddings):
    args = arg_parse()
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float().to('cuda')

    # z = embeddings.cpu().detach().numpy()
    # z = np.transpose(z)
    # c = np.cov(z)
    # _, singular_values, _ = np.linalg.svd(c)

    # Step 1: Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    # Step 2: Calculate covariance matrix
    embeddings_np = embeddings.cpu().detach().numpy()
    cov_matrix = np.cov(embeddings_np, rowvar=False)
    
    # Step 3: Singular Value Decomposition (SVD)
    _, singular_values, _ = np.linalg.svd(cov_matrix)
    
    # Step 4: Plot singular value distribution
    plt.plot(singular_values)
    plt.yscale('log') 
    plt.title("Singular Value Distribution")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (Log)")
    plt.savefig(f"singular_value_distribution_{args.DS}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Step 5: Log singular values for inspection
    print("Singular Values:", singular_values)
    return singular_values

