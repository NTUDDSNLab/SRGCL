import numpy as np
from collections import defaultdict

def parse_and_summarize(log_path):
    # Structure: {(dataset, model): [auc1, auc2, ..., aucN]}
    results = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            dataset, seed, model_path, auc_str = parts
            try:
                auc = float(auc_str)
            except ValueError:
                continue
            key = (dataset, model_path)
            results[key].append(auc)

    print(f"{'Dataset':<10}  {'Model':<30}  {'Count':>5}  {'Mean (%)':>8}  {'Std (%)':>8}")
    print("-" * 70)

    for (dataset, model), auc_list in sorted(results.items()):
        count = len(auc_list)
        if count != 10:
            print(f"⚠️ WARNING: {dataset}, {model} only has {count} samples (expected 10).")

        arr  = np.array(auc_list) * 100
        mean = np.round(arr.mean(), 2)
        std  = np.round(arr.std(ddof=0),  2)
        print(f"{dataset:<10}  {model:<30}  {count:5d}  {mean:8.2f}  {std:8.2f}")

if __name__ == "__main__":
    parse_and_summarize("result.log")