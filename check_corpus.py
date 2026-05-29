import argparse
import pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_save_path", required=True)
    ap.add_argument("--train_cutoff", type=int, default=2020)
    ap.add_argument("--test_year_max", type=int, default=2021)
    args = ap.parse_args()

    d = pickle.load(open(args.data_save_path, "rb"))
    times = d["times"]
    theta = d["theta"]
    K = theta.shape[1]

    print(f"\n{'='*45}")
    print(f"CORPUS DIAGNOSTIC  (K={K}, D={len(times):,})")
    print(f"{'='*45}")

    print("\nPapers per year:")
    for y in range(int(times.min()), int(times.max()) + 1):
        n = int(((times >= y) & (times < y + 1)).sum())
        bar = "#" * (n // 200)
        print(f"  {y}: {n:6,}  {bar}")

    train_end = float(args.train_cutoff + 1)
    test_end = float(args.test_year_max + 1)
    n_train = int((times < train_end).sum())
    n_test = int(((times >= train_end) & (times < test_end)).sum())
    print(f"\nTrain (< {args.train_cutoff+1}): {n_train:,}")
    print(f"Test  ({args.train_cutoff+1}-{args.test_year_max}): {n_test:,}")
    print(f"Train/Test ratio: {n_train/max(n_test,1):.2f}  "
          f"({'GOOD: train > test' if n_train > n_test else 'WARNING: train < test'})")

    n_sample = min(500, len(theta))
    idx = np.random.RandomState(0).choice(len(theta), n_sample, replace=False)
    sample = theta[idx]
    norms = np.linalg.norm(sample, axis=1, keepdims=True) + 1e-10
    cos = (sample @ sample.T) / (norms @ norms.T)
    mean_cos = cos[np.triu_indices(n_sample, 1)].mean()
    print(f"\nMean pairwise topic cosine: {mean_cos:.4f}")
    if mean_cos < 0.35:
        print("  GOOD: topics discriminate well (citation ranking should improve)")
    elif mean_cos < 0.45:
        print("  OK: moderate discrimination")
    else:
        print("  WARNING: topics still homogeneous — consider higher K or broader corpus")
    print()


if __name__ == "__main__":
    main()
