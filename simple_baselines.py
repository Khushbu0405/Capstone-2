import json
import logging
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def _jsd(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q))


def _cosine(a, b, eps=1e-10):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-10)


def evaluate_baseline(name, theta, times, beta, vocab, cit_mat,
                      test_idx, config,
                      predict_fn, rank_fn):
    K_vals = config.get("recall_at_k", [5, 10, 20])

    jsd_scores = []
    cosine_scores = []
    recall_scores = {k: [] for k in K_vals}
    mrr_scores = []
    auc_scores = []
    n_no_cite = 0

    for d in tqdm(test_idx, desc=f"  {name}", unit="paper"):
        t_d = times[d]
        theta_d = theta[d]
        hist_idx = np.where(times < t_d)[0]
        if len(hist_idx) == 0:
            continue

        # Topic prediction
        pi_k = predict_fn(t_d)
        jsd_scores.append(_jsd(pi_k, theta_d))
        cosine_scores.append(_cosine(pi_k, theta_d))

        # Citation ranking
        cited_set = cit_mat.get(d, set())
        if not cited_set:
            n_no_cite += 1
            continue

        scores = rank_fn(d, hist_idx)
        cited_positions = set()
        for pos, h in enumerate(hist_idx):
            if h in cited_set:
                cited_positions.add(pos)
        if not cited_positions:
            n_no_cite += 1
            continue

        relevance = np.zeros(len(hist_idx))
        for pos in cited_positions:
            relevance[pos] = 1.0

        ranked = np.argsort(scores)[::-1]

        for k in K_vals:
            top_k = set(ranked[:k])
            hits = len(cited_positions & top_k)
            recall_scores[k].append(hits / len(cited_positions))

        for rank, pos in enumerate(ranked, start=1):
            if pos in cited_positions:
                mrr_scores.append(1.0 / rank)
                break

        if relevance.sum() > 0 and relevance.sum() < len(relevance):
            try:
                auc_scores.append(roc_auc_score(relevance, scores))
            except ValueError:
                pass

    # Word prediction
    target_year = config.get("prediction_year", 2020)
    topn = config.get("prediction_topn", 30)
    bin_width = config.get("prediction_bin_width", 1.0)
    V = len(vocab)

    t_start = float(target_year)
    t_end = float(target_year + bin_width)
    target_idx = np.where((times >= t_start) & (times < t_end))[0]

    pred_metrics = {}
    if len(target_idx) > 0:
        from topic_model.lda_trainer import TextPreprocessor
        word2id = {w: i for i, w in enumerate(vocab)}
        actual_probs = np.zeros(V, dtype=np.float64)
        preprocessor = TextPreprocessor()
        from pathlib import Path
        t_star = float(target_year + 0.5 * bin_width)
        pi_k_pred = predict_fn(t_star)
        pred_probs = pi_k_pred @ beta
        pred_probs /= pred_probs.sum() + 1e-10

        pred_metrics["pred_pi_k"] = pi_k_pred
        pred_metrics["pred_word_probs"] = pred_probs

    metrics = {}
    if jsd_scores:
        metrics["topic_JSD_mean"] = float(np.mean(jsd_scores))
        metrics["topic_JSD_std"] = float(np.std(jsd_scores))
        metrics["topic_Cosine_mean"] = float(np.mean(cosine_scores))
        metrics["topic_Cosine_std"] = float(np.std(cosine_scores))
    if mrr_scores:
        for k in K_vals:
            metrics[f"Recall@{k}"] = float(np.mean(recall_scores[k]))
        metrics["MRR"] = float(np.mean(mrr_scores))
    if auc_scores:
        metrics["AUC-ROC"] = float(np.mean(auc_scores))

    metrics["_pred_metrics"] = pred_metrics
    return metrics


def compute_word_prediction_metrics(name, pred_probs, papers, times, vocab,
                                     target_year, topn, bin_width):
    
    from topic_model.lda_trainer import TextPreprocessor

    V = len(vocab)
    t_start = float(target_year)
    t_end = float(target_year + bin_width)
    target_idx = np.where((times >= t_start) & (times < t_end))[0]

    if len(target_idx) == 0:
        return {}

    word2id = {w: i for i, w in enumerate(vocab)}
    actual_probs = np.zeros(V, dtype=np.float64)
    preprocessor = TextPreprocessor()
    for text in papers.iloc[target_idx]["text"].fillna("").tolist():
        for tok in preprocessor.preprocess(text):
            wid = word2id.get(tok)
            if wid is not None:
                actual_probs[wid] += 1.0
    actual_probs /= actual_probs.sum() + 1e-10

    pred_top = pred_probs.argsort()[::-1][:topn]
    actual_top = actual_probs.argsort()[::-1][:topn]
    overlap = len(set(pred_top) & set(actual_top)) / max(topn, 1)

    return {
        "pred_actual_jsd": _jsd(pred_probs, actual_probs),
        "pred_actual_cosine": _cosine(pred_probs, actual_probs),
        "pred_actual_topn_overlap": float(overlap),
        "predicted_top_words": [vocab[i] for i in pred_top],
        "actual_top_words": [vocab[i] for i in actual_top],
    }


def run_static_lda(data, config, test_idx):
    log.info("\n" + "=" * 60)
    log.info("BASELINE 1: Static LDA (no temporal modeling)")
    log.info("=" * 60)

    theta = data["theta"]
    times = data["times"]
    beta = data["beta"]
    vocab = data["vocab"]
    cit_mat = data["citation_matrix"]
    window = config.get("context_window", 2.0)

    def predict_fn(t_d):
        mask = (times >= t_d - window) & (times < t_d)
        recent = np.where(mask)[0]
        if len(recent) == 0:
            return theta.mean(axis=0)
        mean_theta = theta[recent].mean(axis=0)
        return mean_theta / (mean_theta.sum() + 1e-10)

    def rank_fn(d, hist_idx):
        theta_d = theta[d]
        theta_hist = theta[hist_idx]
        # Cosine similarity
        norms_d = np.linalg.norm(theta_d) + 1e-10
        norms_h = np.linalg.norm(theta_hist, axis=1) + 1e-10
        return (theta_hist @ theta_d) / (norms_h * norms_d)

    metrics = evaluate_baseline(
        "Static LDA", theta, times, beta, vocab, cit_mat,
        test_idx, config, predict_fn, rank_fn
    )

    # Word prediction
    t_star = float(config.get("prediction_year", 2020)) + 0.5
    pi_k = predict_fn(t_star)
    pred_probs = pi_k @ beta
    pred_probs /= pred_probs.sum() + 1e-10
    word_metrics = compute_word_prediction_metrics(
        "Static LDA", pred_probs, data["papers"], times, vocab,
        config.get("prediction_year", 2020),
        config.get("prediction_topn", 30),
        config.get("prediction_bin_width", 1.0),
    )
    metrics.update(word_metrics)
    del metrics["_pred_metrics"]

    _print_metrics("Static LDA", metrics, config)
    return metrics


def run_uniform_hawkes(data, config, test_idx):
    log.info("\n" + "=" * 60)
    log.info("BASELINE 2: LDA + Uniform Hawkes (no attention)")
    log.info("=" * 60)

    theta = data["theta"]
    times = data["times"]
    beta = data["beta"]
    vocab = data["vocab"]
    cit_mat = data["citation_matrix"]

    
    k_wb = config.get("weibull_k", 1.5)
    lam_wb = config.get("weibull_lam", 2.0)
    mu = config.get("mu_init", 0.1)

    def weibull_pdf(dt):
        dt = np.clip(dt, 1e-8, None)
        x = dt / lam_wb
        return (k_wb / lam_wb) * (x ** (k_wb - 1)) * np.exp(-(x ** k_wb))

    def predict_fn(t_d):
        hist_mask = times < t_d
        hist_idx = np.where(hist_mask)[0]
        if len(hist_idx) == 0:
            return np.ones(theta.shape[1]) / theta.shape[1]

        dt = t_d - times[hist_idx]
        f_dt = weibull_pdf(dt)

        
        N = len(hist_idx)
        uniform_w = np.ones(N) / N
        combined = uniform_w * f_dt
        excitation = theta[hist_idx].T @ combined  # [K]
        lambda_k = mu + excitation
        return softmax(lambda_k)

    def rank_fn(d, hist_idx):
        dt = times[d] - times[hist_idx]
        return weibull_pdf(dt)

    metrics = evaluate_baseline(
        "Uniform Hawkes", theta, times, beta, vocab, cit_mat,
        test_idx, config, predict_fn, rank_fn
    )

    t_star = float(config.get("prediction_year", 2020)) + 0.5
    pi_k = predict_fn(t_star)
    pred_probs = pi_k @ beta
    pred_probs /= pred_probs.sum() + 1e-10
    word_metrics = compute_word_prediction_metrics(
        "Uniform Hawkes", pred_probs, data["papers"], times, vocab,
        config.get("prediction_year", 2020),
        config.get("prediction_topn", 30),
        config.get("prediction_bin_width", 1.0),
    )
    metrics.update(word_metrics)
    del metrics["_pred_metrics"]

    _print_metrics("Uniform Hawkes", metrics, config)
    return metrics


def run_content_recency(data, config, test_idx):
    log.info("\n" + "=" * 60)
    log.info("BASELINE 3: Content + Recency (no learning)")
    log.info("=" * 60)

    theta = data["theta"]
    times = data["times"]
    beta = data["beta"]
    vocab = data["vocab"]
    cit_mat = data["citation_matrix"]
    window = config.get("context_window", 2.0)
    decay = config.get("decay_rate", 1.0)

    def predict_fn(t_d):
        mask = (times >= t_d - window) & (times < t_d)
        recent = np.where(mask)[0]
        if len(recent) == 0:
            return theta.mean(axis=0)
        dt = t_d - times[recent]
        weights = np.exp(-decay * dt)
        weights /= weights.sum() + 1e-10
        mean_theta = (weights[:, None] * theta[recent]).sum(axis=0)
        return mean_theta / (mean_theta.sum() + 1e-10)

    def rank_fn(d, hist_idx):
        theta_d = theta[d]
        theta_hist = theta[hist_idx]

        # Cosine similarity
        norms_d = np.linalg.norm(theta_d) + 1e-10
        norms_h = np.linalg.norm(theta_hist, axis=1) + 1e-10
        cos_sim = (theta_hist @ theta_d) / (norms_h * norms_d)

        # Time decay
        dt = times[d] - times[hist_idx]
        time_weight = np.exp(-decay * dt)

        return cos_sim * time_weight

    metrics = evaluate_baseline(
        "Content+Recency", theta, times, beta, vocab, cit_mat,
        test_idx, config, predict_fn, rank_fn
    )

    t_star = float(config.get("prediction_year", 2020)) + 0.5
    pi_k = predict_fn(t_star)
    pred_probs = pi_k @ beta
    pred_probs /= pred_probs.sum() + 1e-10
    word_metrics = compute_word_prediction_metrics(
        "Content+Recency", pred_probs, data["papers"], times, vocab,
        config.get("prediction_year", 2020),
        config.get("prediction_topn", 30),
        config.get("prediction_bin_width", 1.0),
    )
    metrics.update(word_metrics)
    del metrics["_pred_metrics"]

    _print_metrics("Content+Recency", metrics, config)
    return metrics

def _print_metrics(name, metrics, config):
    K_vals = config.get("recall_at_k", [5, 10, 20])
    log.info(f"\n{'─' * 50}")
    log.info(f"  {name} RESULTS")
    log.info(f"{'─' * 50}")
    log.info(f"  Topic JSD  (↓)  : {metrics.get('topic_JSD_mean', 'N/A')}")
    log.info(f"  Topic Cos  (↑)  : {metrics.get('topic_Cosine_mean', 'N/A')}")
    for k in K_vals:
        log.info(f"  Recall@{k:<3d} (↑)  : {metrics.get(f'Recall@{k}', 'N/A')}")
    log.info(f"  MRR        (↑)  : {metrics.get('MRR', 'N/A')}")
    log.info(f"  AUC-ROC    (↑)  : {metrics.get('AUC-ROC', 'N/A')}")
    if "pred_actual_jsd" in metrics:
        log.info(f"  Pred JSD   (↓)  : {metrics['pred_actual_jsd']:.4f}")
        log.info(f"  Pred Cos   (↑)  : {metrics['pred_actual_cosine']:.4f}")
        log.info(f"  Top-N Ovlp (↑)  : {metrics['pred_actual_topn_overlap']:.4f}")
    log.info(f"{'─' * 50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_save_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/simple_baselines")
    parser.add_argument("--train_cutoff", type=int, default=2019)
    parser.add_argument("--test_year_max", type=int, default=2021)
    parser.add_argument("--context_window", type=float, default=2.0)
    parser.add_argument("--decay_rate", type=float, default=1.0)
    parser.add_argument("--weibull_k", type=float, default=1.5)
    parser.add_argument("--weibull_lam", type=float, default=2.0)
    parser.add_argument("--prediction_year", type=int, default=2020)
    parser.add_argument("--prediction_topn", type=int, default=30)
    args = parser.parse_args()

    config = vars(args)
    config["recall_at_k"] = [5, 10, 20]
    config["prediction_bin_width"] = 1.0
    config["mu_init"] = 0.1

    log.info(f"Loading data from {args.data_save_path}")
    with open(args.data_save_path, "rb") as f:
        data = pickle.load(f)

    times = data["times"]
    train_end = float(args.train_cutoff + 1)
    test_end = float(args.test_year_max + 1)
    test_idx = np.where((times >= train_end) & (times < test_end))[0]
    log.info(f"Test papers: {len(test_idx):,}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Static LDA
    m1 = run_static_lda(data, config, test_idx)
    all_results["static_lda"] = m1

    # 2. Uniform Hawkes
    m2 = run_uniform_hawkes(data, config, test_idx)
    all_results["uniform_hawkes"] = m2

    # 3. Content + Recency
    m3 = run_content_recency(data, config, test_idx)
    all_results["content_recency"] = m3

    log.info("\n" + "=" * 80)
    log.info("  SIMPLE BASELINES COMPARISON")
    log.info("=" * 80)

    header = f"{'Metric':<26} | {'Static LDA':>12} | {'Unif Hawkes':>12} | {'Cont+Recency':>12}"
    log.info(header)
    log.info("─" * 70)

    rows = [
        ("Topic JSD ↓", "topic_JSD_mean"),
        ("Topic Cosine ↑", "topic_Cosine_mean"),
        ("Recall@5 ↑", "Recall@5"),
        ("Recall@10 ↑", "Recall@10"),
        ("Recall@20 ↑", "Recall@20"),
        ("MRR ↑", "MRR"),
        ("AUC-ROC ↑", "AUC-ROC"),
        ("Pred JSD ↓", "pred_actual_jsd"),
        ("Pred Cosine ↑", "pred_actual_cosine"),
        ("Top-30 Overlap ↑", "pred_actual_topn_overlap"),
    ]

    for label, key in rows:
        vals = []
        for name in ["static_lda", "uniform_hawkes", "content_recency"]:
            v = all_results[name].get(key)
            vals.append(f"{v:.4f}" if isinstance(v, float) else "—")
        log.info(f"{label:<26} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12}")

    log.info("─" * 70)

    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            k: v for k, v in m.items()
            if isinstance(v, (int, float, str, list))
        }

    with open(out_dir / "simple_baselines_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"\nResults saved → {out_dir / 'simple_baselines_metrics.json'}")


if __name__ == "__main__":
    main()
