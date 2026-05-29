import os
import sys
import json
import logging
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from gensim.models.coherencemodel import CoherenceModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DSNTM_CONFIG = {
    "data_save_path": None,
    "year_min": 2014,
    "year_max": 2021,
    "train_cutoff": 2020,

    "K": 30,
    "emb_dim": 300,
    "t_hidden": 800,
    "eta_hidden": 200,
    "eta_nlayers": 3,
    "n_heads": 10,
    "delta": 0.005,
    "enc_drop": 0.0,
    "train_embeddings": True,
    "bin_width": 2,

    "use_citation": True,
    "citation_weight": 1.0,
    "min_citation_count": 4,

    "epochs": 30,
    "batch_size": 512,
    "lr": 0.0006,
    "lr_factor": 4.0,
    "weight_decay": 1.2e-6,
    "clip": 2.0,
    "anneal_lr": True,
    "nonmono": 10,
    "bow_norm": True,

    "prediction_year": 2020,
    "prediction_topn": 30,
    "prediction_bin_width": 1.0,

    "device": "cuda",
    "output_dir": None,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train DSNTM baseline")
    p.add_argument("--data_save_path", type=str, default=None)
    p.add_argument("--K", type=int, default=DSNTM_CONFIG["K"])
    p.add_argument("--emb_dim", type=int, default=DSNTM_CONFIG["emb_dim"])
    p.add_argument("--epochs", type=int, default=DSNTM_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=DSNTM_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=DSNTM_CONFIG["lr"])
    p.add_argument("--bin_width", type=int, default=DSNTM_CONFIG["bin_width"])
    p.add_argument("--device", type=str, default=DSNTM_CONFIG["device"])
    p.add_argument("--citation_weight", type=float, default=DSNTM_CONFIG["citation_weight"])
    p.add_argument("--no_citation", action="store_true")
    p.add_argument("--frozen_beta", action="store_true",
                    help="Use frozen LDA β instead of learned time-varying β. "
                         "Makes comparison with Hawkes model fair.")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    config = DSNTM_CONFIG.copy()
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    if args.no_citation:
        config["use_citation"] = False
    config["frozen_beta"] = args.frozen_beta

    return config


def _jsd(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def _cosine(a, b, eps=1e-10):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def train_dsntm(config):
    from dsntm_model import DSNTMBaseline
    from dsntm_data import build_dsntm_data

    data_path = config.get("data_save_path")
    if data_path is None:
        candidates = [
            Path("outputs/processed_data_2014_2021_transformer.pkl"),
            Path("../outputs/processed_data_2014_2021_transformer.pkl"),
        ]
        for c in candidates:
            if c.exists():
                data_path = str(c)
                break
        if data_path is None:
            raise FileNotFoundError(
                "Cannot find processed data pickle. "
                "Run train.py first or specify --data_save_path."
            )

    log.info(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        hawkes_data = pickle.load(f)

    papers = hawkes_data["papers"]
    times = hawkes_data["times"]
    vocab = hawkes_data["vocab"]
    cit_mat = hawkes_data.get("citation_matrix", {})

    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        log.warning("CUDA not available, using CPU")

    log.info("Building DSNTM data structures …")
    dsntm_data = build_dsntm_data(
        papers, times, vocab, cit_mat,
        year_min=config["year_min"],
        year_max=config["year_max"],
        bin_width=config["bin_width"],
        train_cutoff=config["train_cutoff"],
        device=device,
    )

    T = dsntm_data["T"]
    V = dsntm_data["V"]
    K = config["K"]

    frozen_beta = None
    if config.get("frozen_beta", False):
        beta_lda = hawkes_data.get("beta")
        if beta_lda is not None:
            frozen_beta = beta_lda
            log.info(f"Using FROZEN LDA β: shape {beta_lda.shape}")
        else:
            log.warning("--frozen_beta requested but no beta in data pickle")

    model = DSNTMBaseline(
        K=K, V=V, T=T,
        emb_dim=config["emb_dim"],
        t_hidden=config["t_hidden"],
        eta_hidden=config["eta_hidden"],
        eta_nlayers=config["eta_nlayers"],
        n_heads=config["n_heads"],
        delta=config["delta"],
        enc_drop=config["enc_drop"],
        citation=config["use_citation"],
        citation_weight=config["citation_weight"],
        min_citation_count=config.get("min_citation_count", 4),
        train_embeddings=config["train_embeddings"],
        frozen_beta=frozen_beta,
        device=device,
    ).to(device)

    if config["use_citation"] and "citation_by_time" in dsntm_data:
        log.info("Loading citation data into DSNTM model …")
        # Build bow_by_time from training docs
        bow = dsntm_data["bow"]
        time_ids = dsntm_data["time_ids"]
        train_idx = dsntm_data["train_idx"]
        val_idx = dsntm_data["val_idx"]
        train_val_set = set(train_idx.tolist()) | set(val_idx.tolist())

        bow_by_time_all = {}
        for t in range(T):
            mask = []
            for d_idx in range(len(time_ids)):
                if int(time_ids[d_idx]) == t and d_idx in train_val_set:
                    mask.append(d_idx)
            if mask:
                bow_by_time_all[t] = bow[mask]
            else:
                bow_by_time_all[t] = torch.zeros(0, V, device=torch.device(device))

        model.update_citation_data(
            dsntm_data["citation_by_time"],
            bow_by_time_all,
        )
        log.info("  Citation data wired into model ✓")
    else:
        if config["use_citation"]:
            log.warning("Citation enabled but no citation_by_time in data — "
                        "citation loss will be 0")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    train_idx = dsntm_data["train_idx"]
    bow = dsntm_data["bow"]
    time_ids = dsntm_data["time_ids"]
    rnn_inp = dsntm_data.get("train_rnn_inp", dsntm_data["rnn_inp"])
    num_docs_train = len(train_idx)

    log.info(f"\nTraining DSNTM: K={K}, T={T}, V={V}, D_train={num_docs_train}")
    log.info(f"  Epochs={config['epochs']}, BS={config['batch_size']}, "
             f"LR={config['lr']}, Citation={config['use_citation']}")

    best_ppl = float("inf")
    best_state = None
    all_ppls = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        perm = np.random.permutation(train_idx)
        batches = [
            perm[i: i + config["batch_size"]]
            for i in range(0, len(perm), config["batch_size"])
        ]

        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_cit = 0.0
        n_batches = 0

        for batch_idx in batches:
            optimizer.zero_grad()

            batch_bow = bow[batch_idx]
            batch_times = time_ids[batch_idx]
            sums = batch_bow.sum(1, keepdim=True).clamp(min=1)
            if config["bow_norm"]:
                norm_bow = batch_bow / sums
            else:
                norm_bow = batch_bow

            nelbo, nll, kl_eta, kl_theta, kl_alpha, cit_loss = model(
                batch_bow, norm_bow, batch_times, rnn_inp, num_docs_train
            )

            nelbo.backward()
            if config["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["clip"]
                )
            optimizer.step()

            epoch_loss += nelbo.item()
            epoch_nll += nll.item()
            epoch_cit += cit_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cit = epoch_cit / max(n_batches, 1)

        val_ppl = model.compute_perplexity(
            dsntm_data["val_bow_by_time"],
            dsntm_data.get("val_rnn_inp", rnn_inp),
        )
        all_ppls.append(val_ppl)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (config["anneal_lr"]
                and len(all_ppls) > config["nonmono"]
                and val_ppl > min(all_ppls[:-config["nonmono"]])
                and optimizer.param_groups[0]["lr"] > 1e-5):
            optimizer.param_groups[0]["lr"] /= config["lr_factor"]
            log.info(
                f"  LR reduced to {optimizer.param_groups[0]['lr']:.6f}"
            )

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:3d}/{config['epochs']} | "
                f"NELBO {avg_loss:.1f} | "
                f"Cit {avg_cit:.1f} | "
                f"Val PPL {val_ppl:.1f} | "
                f"Best PPL {best_ppl:.1f}"
            )

    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )
    model.eval()
    log.info(f"\nTraining complete. Best validation PPL: {best_ppl:.1f}")

    return model, dsntm_data, hawkes_data


def evaluate_dsntm(model, dsntm_data, hawkes_data, config):
    from topic_model.lda_trainer import TextPreprocessor

    log.info("\n" + "=" * 60)
    log.info("DSNTM BASELINE EVALUATION")
    log.info("=" * 60)

    vocab = hawkes_data["vocab"]
    times = hawkes_data["times"]
    papers = hawkes_data["papers"]
    K = config["K"]

    metrics = {}

    test_ppl = model.compute_perplexity(
        dsntm_data["test_bow_by_time"],
        dsntm_data.get("val_rnn_inp", dsntm_data["rnn_inp"]),
    )
    metrics["perplexity"] = test_ppl
    log.info(f"  Test Perplexity   : {test_ppl:.1f}")

    diversity = model.get_topic_diversity(top_n=25)
    metrics["diversity"] = diversity
    log.info(f"  Topic Diversity   : {diversity:.4f}")

    target_year = config.get("prediction_year", 2020)
    topn = config.get("prediction_topn", 30)
    bin_width_pred = config.get("prediction_bin_width", 1.0)
    V = len(vocab)

    t_start = float(target_year)
    t_end = float(target_year + bin_width_pred)
    target_idx = np.where((times >= t_start) & (times < t_end))[0]

    if len(target_idx) > 0:
        word2id = {w: i for i, w in enumerate(vocab)}
        actual_probs = np.zeros(V, dtype=np.float64)
        preprocessor = TextPreprocessor()
        for text in papers.iloc[target_idx]["text"].fillna("").tolist():
            for token in preprocessor.preprocess(text):
                wid = word2id.get(token)
                if wid is not None:
                    actual_probs[wid] += 1.0
        actual_probs /= actual_probs.sum() + 1e-10

        rnn_inp = dsntm_data.get("train_rnn_inp", dsntm_data["rnn_inp"])
        pred_probs, pi_k = model.predict_word_distribution(rnn_inp)

        pred_top = pred_probs.argsort()[::-1][:topn]
        actual_top = actual_probs.argsort()[::-1][:topn]
        overlap = len(set(pred_top) & set(actual_top)) / max(topn, 1)

        pred_jsd = _jsd(pred_probs, actual_probs)
        pred_cos = _cosine(pred_probs, actual_probs)

        metrics["pred_actual_year"] = int(target_year)
        metrics["pred_actual_topn"] = int(topn)
        metrics["pred_actual_jsd"] = pred_jsd
        metrics["pred_actual_cosine"] = pred_cos
        metrics["pred_actual_topn_overlap"] = float(overlap)
        metrics["predicted_top_words"] = [vocab[i] for i in pred_top]
        metrics["actual_top_words"] = [vocab[i] for i in actual_top]

        log.info(f"\n  PREDICTED VS ACTUAL WORDS ({target_year})")
        log.info(f"  JSD             : {pred_jsd:.4f}  (↓ better)")
        log.info(f"  Cosine          : {pred_cos:.4f}  (↑ better)")
        log.info(f"  Top-{topn} overlap : {overlap:.4f}")
    else:
        log.warning(f"No documents found for target year {target_year}")

    try:
        lda_dict = hawkes_data.get("lda_dictionary")
        if lda_dict is not None:
            beta_all = model.get_beta_all_times()
            coherence_scores = []
            for t in range(dsntm_data["T"]):
                topic_words_list = []
                for k in range(K):
                    top_ids = beta_all[t, k].argsort()[-10:][::-1]
                    topic_words_list.append([vocab[i] for i in top_ids])

                ys, ye = dsntm_data["bin_edges"][t]
                bin_idx = np.where((times >= ys) & (times < ye))[0]
                if len(bin_idx) == 0:
                    continue

                preprocessor = TextPreprocessor()
                tokenized = [
                    [tok for tok in preprocessor.preprocess(text)
                     if tok in lda_dict.token2id]
                    for text in papers.iloc[bin_idx]["text"].fillna("").tolist()
                ]
                tokenized = [doc for doc in tokenized if doc]
                if len(tokenized) < 10:
                    continue

                cm = CoherenceModel(
                    topics=topic_words_list,
                    texts=tokenized,
                    dictionary=lda_dict,
                    coherence="c_npmi",
                    topn=10,
                )
                score = cm.get_coherence()
                if np.isfinite(score):
                    coherence_scores.append(score)

            if coherence_scores:
                mean_coh = float(np.mean(coherence_scores))
                std_coh = float(np.std(coherence_scores))
                metrics["coherence_npmi_mean"] = mean_coh
                metrics["coherence_npmi_std"] = std_coh
                log.info(f"\n  NPMI Coherence  : {mean_coh:.4f} ± {std_coh:.4f}")
    except Exception as e:
        log.warning(f"Coherence computation failed: {e}")

    last_t = dsntm_data["T"] - 1
    topic_words = model.get_top_words_per_topic(vocab, t=last_t, top_n=8)
    log.info(f"\n  === DSNTM Topics (time bin {last_t}) ===")
    for k, words in enumerate(topic_words):
        log.info(f"    Topic {k:2d}: {', '.join(words)}")

    log.info(f"\n{'─' * 50}")
    log.info("DSNTM BASELINE RESULTS SUMMARY")
    log.info(f"{'─' * 50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info(f"  {k:30s} : {v:.4f}")
        elif isinstance(v, int):
            log.info(f"  {k:30s} : {v}")
    log.info(f"{'─' * 50}\n")

    return metrics


def main():
    config = parse_args()

    if config["output_dir"] is None:
        config["output_dir"] = str(
            Path(__file__).resolve().parent / "outputs" / "dsntm_baseline"
        )
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, dsntm_data, hawkes_data = train_dsntm(config)

    model_path = out_dir / "dsntm_baseline_model.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved → {model_path}")

    metrics = evaluate_dsntm(model, dsntm_data, hawkes_data, config)

    metrics_path = out_dir / "dsntm_baseline_metrics.json"
    serializable = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, list))
    }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    config_path = out_dir / "dsntm_baseline_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return metrics


if __name__ == "__main__":
    main()
