#!/usr/bin/env python3

import json
import argparse
from pathlib import Path


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hawkes_metrics",
        type=str,
        default="outputs/eval_metrics_v2.json", 
    )
    parser.add_argument(
        "--dsntm_metrics",
        type=str,
        default="outputs/dsntm_baseline/dsntm_baseline_metrics.json",
    )
    parser.add_argument(
        "--dsntm_frozen_metrics",
        type=str,
        default="outputs/dsntm_frozen_beta/dsntm_baseline_metrics.json", 
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/three_way_model_comparison.json",
    )
    args = parser.parse_args()

    hawkes = load_metrics(args.hawkes_metrics) if Path(args.hawkes_metrics).exists() else {}
    dsntm = load_metrics(args.dsntm_metrics) if Path(args.dsntm_metrics).exists() else {}
    dsntm_frozen = load_metrics(args.dsntm_frozen_metrics) if Path(args.dsntm_frozen_metrics).exists() else {}

    if not hawkes:
        print(f"[WARNING] Hawkes metrics not found at {args.hawkes_metrics}")
    if not dsntm:
        print(f"[WARNING] Standard DSNTM metrics not found at {args.dsntm_metrics}")
    if not dsntm_frozen:
        print(f"[WARNING] Frozen-β DSNTM metrics not found at {args.dsntm_frozen_metrics}")

    rows = [
        ("Perplexity",             None,                      "perplexity",             "↓"),
        ("Topic Diversity",        None,                      "diversity",              "↑"),
        ("NPMI Coherence",         "hawkes_coherence_mean",   "coherence_npmi_mean",    "↑"),
        ("Pred vs Actual JSD",     "pred_actual_jsd",         "pred_actual_jsd",        "↓"),
        ("Pred vs Actual Cosine",  "pred_actual_cosine",      "pred_actual_cosine",     "↑"),
        ("Top-N Word Overlap",     "pred_actual_topn_overlap","pred_actual_topn_overlap","↑"),
        ("Topic JSD (mean)",       "topic_JSD_mean",          None,                     "↓"),
        ("Topic Cosine (mean)",    "topic_Cosine_mean",       None,                     "↑"),
        ("Recall@5",               "Recall@5",                None,                     "↑"),
        ("Recall@10",              "Recall@10",               None,                     "↑"),
        ("Recall@20",              "Recall@20",               None,                     "↑"),
        ("MRR",                    "MRR",                     None,                     "↑"),
        ("AUC-ROC",                "AUC-ROC",                 None,                     "↑"),
    ]
    header = f"{'Metric':<24} | {'Hawkes (V2)':>13} | {'DSNTM (Dyn)':>12} | {'DSNTM (Froz)':>13} | {'Best':>12} | {'Dir':>3}"
    sep = "─" * len(header)

    print("\n" + sep)
    print("  MODEL COMPARISON: Hawkes (V2) vs DSNTM Baseline variants")
    print(sep)
    print(header)
    print(sep)

    comparison = {}
    for name, h_key, d_key, direction in rows:
        h_val = hawkes.get(h_key) if h_key else None
        d_val = dsntm.get(d_key) if d_key else None
        df_val = dsntm_frozen.get(d_key) if d_key else None  

        h_str = f"{h_val:.4f}" if isinstance(h_val, (int, float)) else "     —"
        d_str = f"{d_val:.4f}" if isinstance(d_val, (int, float)) else "   —"
        df_str = f"{df_val:.4f}" if isinstance(df_val, (int, float)) else "     —"

        # Multi-model evaluation logic
        candidates = {}
        if isinstance(h_val, (int, float)): candidates["Hawkes"] = h_val
        if isinstance(d_val, (int, float)): candidates["DSNTM(Dyn)"] = d_val
        if isinstance(df_val, (int, float)): candidates["DSNTM(Froz)"] = df_val

        best = "     —"
        if candidates:
            if direction == "↓":
                best = min(candidates, key=candidates.get)
            else:
                best = max(candidates, key=candidates.get)

        print(f"{name:<24} | {h_str:>13} | {d_str:>12} | {df_str:>13} | {best:>12} | {direction:>3}")
        
        comparison[name] = {
            "hawkes": h_val,
            "dsntm_dynamic": d_val,
            "dsntm_frozen": df_val,
            "best": best,
            "direction": direction,
        }

    print(sep)
    h_words = hawkes.get("predicted_top_words", [])
    d_words = dsntm.get("predicted_top_words", [])
    df_words = dsntm_frozen.get("predicted_top_words", [])
    a_words = hawkes.get("actual_top_words", []) or dsntm.get("actual_top_words", [])

    if h_words or d_words or df_words:
        print(f"\n{'─' * 85}")
        print("  TOP PREDICTED WORDS COMPARISON")
        print(f"{'─' * 85}")
        n = max(len(h_words), len(d_words), len(df_words), len(a_words))
        n = min(n, 15)
        word_header = f"{'Rank':>4} | {'Hawkes':<18} | {'DSNTM (Dyn)':<18} | {'DSNTM (Froz)':<18} | {'Actual':<18}"
        print(word_header)
        print("─" * len(word_header))
        for i in range(n):
            hw = h_words[i] if i < len(h_words) else "—"
            dw = d_words[i] if i < len(d_words) else "—"
            dfw = df_words[i] if i < len(df_words) else "—"
            aw = a_words[i] if i < len(a_words) else "—"
            print(f"{i+1:>4} | {hw:<18} | {dw:<18} | {dfw:<18} | {aw:<18}")
        print("─" * len(word_header))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nThree-way comparison metrics saved → {out_path}")


if __name__ == "__main__":
    main()