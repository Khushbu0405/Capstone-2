import os
import sys
import logging
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax as scipy_softmax

sys.path.append(os.path.dirname(__file__))
from data.scievo_loader      import load_scievo
from topic_model.lda_trainer import run_lda_pipeline, load_lda_outputs, TextPreprocessor
from hawkes.model            import HawkesTopicModel
from visualize               import visualize_lda

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CONFIG = {
    "year_min":         2014,
    "year_max":         2021,
    "train_cutoff":     2020,   
    "test_year_max":    2021,
    "min_abstract_len": 50,
    "subject_tags":     ["cs.CL", "cs.LG"],   
    "keyword_terms":    [],

    # LDA
    "K":                30,     
    "lda_passes":       25,
    "no_below":         20,
    "no_above":         0.6,
    "keep_n":           8000,
    "use_bigrams":      True,
    "bigram_min_count": 20,
    "bigram_threshold": 10.0,

    "citation_weight":  0.5,    
    "weibull_k":        1.5,
    "weibull_lam":      2.0,    

    "epochs":           50,     
    "batch_size":       32,    
    "lr":               0.005,  
    "grad_clip":        1.0,
    "min_history":      3,

    "max_history":      500,    
    "neg_samples":      50,     
    "focal_gamma":      2.0,    
    "excitation_weight": 0.3,   

    # Evaluation
    "recall_at_k":      [5, 10, 20],
    "context_window":   2.0,
    "decay_rate":       1.0,

    # Predicted vs actual word distribution
    "prediction_year":  2020,
    "prediction_topn":  30,
    "prediction_bin_width": 1.0,

    "use_torch":        True,
    "torch_device":     "cuda",

    # Hawkes coherence 
    "hawkes_coherence_enabled": True,
    "hawkes_coherence_metric": "c_npmi",
    "hawkes_coherence_topn": 20,
    "hawkes_coherence_sample_per_bin": 10000,
    "hawkes_coherence_year_start": None,
    "hawkes_coherence_year_end": None,
    "hawkes_coherence_bin_width": 1.0,

    "lda_save_dir":     str(OUTPUT_DIR / "lda_2014_2021_cscl_cslg_k30"),
    "model_save_path":  str(OUTPUT_DIR / "hawkes_model_v3_cscl_cslg_k30.pkl"),
    "data_save_path":   str(OUTPUT_DIR / "processed_data_2014_2021_cscl_cslg_k30.pkl"),
    "vis_out_dir":      str(OUTPUT_DIR / "visualizations_v3_cscl_cslg_k30"),
}

def prepare_data(config):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if Path(config["data_save_path"]).exists():
        log.info("Loading cached processed data …")
        with open(config["data_save_path"], "rb") as f:
            data = pickle.load(f)

        if "lda_dictionary" not in data:
            try:
                _, _, vocab, lda_model = load_lda_outputs(config["lda_save_dir"])
                data["vocab"] = vocab
                data["lda_dictionary"] = lda_model.id2word
            except Exception as exc:
                log.warning(f"Failed to load LDA dictionary: {exc}")

        cached_k = data.get("beta", np.empty((0,))).shape[0]
        if cached_k and cached_k != config["K"]:
            log.warning(
                f"Cached LDA K={cached_k} does not match config K={config['K']}. "
                "Rebuilding LDA outputs."
            )
            papers = data["papers"]
            beta, theta, vocab, lda_model = run_lda_pipeline(
                papers,
                K=config["K"],
                passes=config["lda_passes"],
                no_below=config["no_below"],
                no_above=config["no_above"],
                keep_n=config["keep_n"],
                save_dir=config["lda_save_dir"],
                load_if_exists=False,
            )

            data["beta"] = beta
            data["theta"] = theta
            data["vocab"] = vocab
            data["lda_dictionary"] = lda_model.id2word

            with open(config["data_save_path"], "wb") as f:
                pickle.dump(data, f)
            log.info(f"Processed data cache updated → {config['data_save_path']}")

        return data

    papers, id_to_idx, citation_dict, citation_matrix = load_scievo(
        year_min=config["year_min"],
        year_max=config["year_max"],
        min_abstract_len=config["min_abstract_len"],
        subject_tags=config.get("subject_tags", []),
        keyword_terms=config.get("keyword_terms", []),
    )

    MAX_PAPERS = 100000  # Limits dataset to a stable 100k memory footprint
    if len(papers) > MAX_PAPERS:
        log.info(f"Broad corpus too large ({len(papers):,} papers). Downsampling to {MAX_PAPERS:,} for RAM stability...")
        
        # Deterministically sample a subset of row indices
        rng = np.random.RandomState(42)
        keep_indices = sorted(rng.choice(len(papers), size=MAX_PAPERS, replace=False))
        keep_set = set(keep_indices)
        
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        #  Filter and slice the papers DataFrame, resetting the index positions
        papers = papers.iloc[keep_indices].reset_index(drop=True)
        
        
        new_citation_matrix = {}
        for old_citing, old_cited_set in citation_matrix.items():
            old_citing = int(old_citing)
            if old_citing in keep_set:
                new_citing = old_to_new[old_citing]
                new_cited_set = {old_to_new[int(c)] for c in old_cited_set if int(c) in keep_set}
                if new_cited_set:
                    new_citation_matrix[new_citing] = new_cited_set
        citation_matrix = new_citation_matrix

    beta, theta, vocab, lda_model = run_lda_pipeline(
        papers,
        K=config["K"],
        passes=config["lda_passes"],
        no_below=config["no_below"],
        no_above=config["no_above"],
        keep_n=config["keep_n"],
        use_bigrams=config.get("use_bigrams", True),
        bigram_min_count=config.get("bigram_min_count", 20),
        bigram_threshold=config.get("bigram_threshold", 10.0),
        save_dir=config["lda_save_dir"],
        load_if_exists=True,
    )

    times = papers["time"].values.astype(np.float64)

    cit_matrix_int = {
        int(idx): {int(j) for j in cited_set}
        for idx, cited_set in citation_matrix.items()
        if cited_set
    }

    data = {
        "papers":          papers,
        "id_to_idx":       id_to_idx,
        "citation_dict":   citation_dict,
        "citation_matrix": cit_matrix_int,
        "beta":            beta,
        "theta":           theta,
        "vocab":           vocab,
        "lda_dictionary":  lda_model.id2word,
        "times":           times,
    }

    with open(config["data_save_path"], "wb") as f:
        pickle.dump(data, f)
    log.info(f"Processed data cached → {config['data_save_path']}")
    return data


def make_split(times, train_cutoff, test_year_max):
    train_end  = float(train_cutoff + 1)
    test_end   = float(test_year_max + 1)

    train_idx = np.where(times <  train_end)[0]
    test_idx  = np.where((times >= train_end) & (times < test_end))[0]

    log.info(f"Train/test split at {train_cutoff}/{train_cutoff+1}:")
    log.info(f"  Train papers : {len(train_idx):,}  "
             f"(time < {train_end:.1f})")
    log.info(f"  Test  papers : {len(test_idx):,}  "
             f"({train_end:.1f} ≤ time < {test_end:.1f})")
    return train_idx, test_idx


def compute_hawkes_coherence_time_bins(model, data, config):
    if not config.get("hawkes_coherence_enabled", True):
        return {}

    metric = config.get("hawkes_coherence_metric", "c_npmi")
    topn = int(config.get("hawkes_coherence_topn", 10))
    sample_per_bin = int(config.get("hawkes_coherence_sample_per_bin", 5000))
    bin_width = float(config.get("hawkes_coherence_bin_width", 1.0))

    year_start = config.get("hawkes_coherence_year_start")
    if year_start is None:
        year_start = config["train_cutoff"] + 1
    year_end = config.get("hawkes_coherence_year_end")
    if year_end is None:
        year_end = config["test_year_max"]

    if year_end < year_start:
        log.warning("Hawkes coherence skipped: invalid year range")
        return {}

    papers = data["papers"]
    times = data["times"]
    beta = data["beta"]
    vocab = data["vocab"]
    dictionary = data.get("lda_dictionary")

    if dictionary is None:
        log.warning("Hawkes coherence skipped: missing LDA dictionary")
        return {}

    preprocessor = TextPreprocessor()
    rng = np.random.RandomState(42)

    bin_scores = {}
    bin_starts = np.arange(year_start, year_end + 1e-9, bin_width)

    for start in bin_starts:
        end = start + bin_width
        idx = np.where((times >= start) & (times < end))[0]
        if len(idx) == 0:
            continue

        if sample_per_bin > 0 and len(idx) > sample_per_bin:
            idx = rng.choice(idx, size=sample_per_bin, replace=False)

        texts = papers.iloc[idx]["text"].fillna("").tolist()
        tokenized = [preprocessor.preprocess(t) for t in texts]
        tokenized = [[tok for tok in doc if tok in dictionary.token2id] for doc in tokenized]
        tokenized = [doc for doc in tokenized if doc]
        if not tokenized:
            continue

        t_star = float(start + 0.5 * bin_width)
        _, pi_k = model.predict_topic_intensity(
            t_star, context_window=config["context_window"]
        )
        word_probs = pi_k @ beta
        top_ids = word_probs.argsort()[::-1][:topn]
        topic_words = [vocab[i] for i in top_ids]

        if metric == "u_mass":
            corpus = [dictionary.doc2bow(doc) for doc in tokenized]
            corpus = [bow for bow in corpus if bow]
            if not corpus:
                continue
            cm = CoherenceModel(
                topics=[topic_words],
                corpus=corpus,
                dictionary=dictionary,
                coherence=metric,
            )
            docs_used = len(corpus)
        else:
            cm = CoherenceModel(
                topics=[topic_words],
                texts=tokenized,
                dictionary=dictionary,
                coherence=metric,
            )
            docs_used = len(tokenized)

        score = cm.get_coherence()
        if not np.isfinite(score):
            bin_key = f"{start:.0f}-{end:.0f}"
            log.warning(
                f"Hawkes coherence {metric} {bin_key}: non-finite score, skipping"
            )
            continue

        bin_key = f"{start:.0f}-{end:.0f}"
        bin_scores[bin_key] = float(score)
        log.info(
            f"Hawkes coherence {metric} {bin_key}: {score:.4f} "
            f"(docs={docs_used:,})"
        )

    if not bin_scores:
        log.warning("Hawkes coherence skipped: no valid bins")
        return {}

    scores = np.array(list(bin_scores.values()), dtype=float)
    return {
        "hawkes_coherence_metric": metric,
        "hawkes_coherence_mean": float(scores.mean()),
        "hawkes_coherence_std": float(scores.std()),
        "hawkes_coherence_by_bin": bin_scores,
    }


def compare_predicted_vs_actual_words(model, data, config):
    target_year = config.get("prediction_year")
    if target_year is None:
        return {}

    topn      = int(config.get("prediction_topn", 30))
    bin_width = float(config.get("prediction_bin_width", 1.0))

    times  = data["times"]
    beta   = data["beta"]
    vocab  = data["vocab"]
    papers = data["papers"]

    t_start = float(target_year)
    t_end   = float(target_year + bin_width)
    idx = np.where((times >= t_start) & (times < t_end))[0]
    if len(idx) == 0:
        log.warning(f"No documents found for prediction year {target_year}.")
        return {}

    word2id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    actual_word_probs = np.zeros(V, dtype=np.float64)
    preprocessor_eval = TextPreprocessor()
    for text in papers.iloc[idx]["text"].fillna("").tolist():
        for w in preprocessor_eval.preprocess(text):
            wid = word2id.get(w)
            if wid is not None:
                actual_word_probs[wid] += 1.0
    actual_word_probs /= actual_word_probs.sum() + 1e-10

    t_star = float(target_year + 0.5 * bin_width)
    _, pi_k = model.predict_topic_intensity(
        t_star, context_window=config["context_window"]
    )
    pred_word_probs = pi_k @ beta
    pred_word_probs = pred_word_probs / (pred_word_probs.sum() + 1e-10)

    pred_top = pred_word_probs.argsort()[::-1][:topn]
    actual_top = actual_word_probs.argsort()[::-1][:topn]
    overlap = len(set(pred_top) & set(actual_top)) / max(topn, 1)

    predicted_top_words = [vocab[i] for i in pred_top]
    actual_top_words = [vocab[i] for i in actual_top]

    pred_words = ", ".join(predicted_top_words[:15])
    actual_words = ", ".join(actual_top_words[:15])
    log.info(f"PREDICTED VS ACTUAL {target_year}")
    log.info(f"  Top-{topn} overlap : {overlap:.4f}")
    log.info(f"  Predicted words   : {pred_words}")
    log.info(f"  Actual words      : {actual_words}")

    return {
        "pred_actual_year": int(target_year),
        "pred_actual_topn": int(topn),
        "pred_actual_jsd": _jsd(pred_word_probs, actual_word_probs),
        "pred_actual_cosine": _cosine(pred_word_probs, actual_word_probs),
        "pred_actual_topn_overlap": float(overlap),
        "predicted_top_words": predicted_top_words,
        "actual_top_words": actual_top_words,
    }


def train(config=None):
    if config is None:
        config = CONFIG

    # Stage 1: data + LDA
    log.info("\n" + "=" * 60)
    log.info("STAGE 1: Loading data and pre-training LDA")
    log.info("=" * 60)

    data = prepare_data(config)

    beta    = data["beta"]
    theta   = data["theta"]
    times   = data["times"]
    vocab   = data["vocab"]
    cit_mat = data["citation_matrix"]

    D = theta.shape[0]
    K = config["K"]
    log.info(f"\nDataset: D={D:,} papers, K={K} topics, V={len(vocab):,}")

    #Visualise LDA outputs
    log.info("\nGenerating LDA visualisations …")
    visualize_lda(
        beta, theta, times, vocab,
        top_n_words=12,
        n_time_bins=40,
        out_dir=config["vis_out_dir"],
    )

    # Stage 2: Hawkes training 
    log.info("\n" + "=" * 60)
    log.info("STAGE 2: Training Hawkes Topic Model (V3)")
    log.info("=" * 60)

    train_idx, test_idx = make_split(
        times,
        config["train_cutoff"],
        config["test_year_max"],
    )

    sorted_order = np.argsort(times)
    train_indices = []
    for d in sorted_order:
        if d not in set(train_idx):
            continue
        n_prev = int(np.sum(times < times[d]))
        if n_prev >= config["min_history"]:
            train_indices.append(d)

    log.info(f"\nTrainable documents (≥{config['min_history']} prev papers): "
             f"{len(train_indices):,}")

    use_torch = bool(config.get("use_torch", False))
    if use_torch:
        import torch
        # ── CHANGED: import V3 model ──
        from hawkes.torch_model_v3 import HawkesTopicModelTorchV3

        device = config.get("torch_device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA not available; falling back to CPU for torch model")
            device = "cpu"

        model = HawkesTopicModelTorchV3(
            K=K,
            beta=beta,
            theta=theta,
            times=times,
            citation_weight=config["citation_weight"],
            weibull_k=config["weibull_k"],
            weibull_lam=config["weibull_lam"],
            context_window=config["context_window"],
            decay_rate=config.get("decay_rate", 1.0),
            max_history=config.get("max_history", 500),
            neg_samples=config.get("neg_samples", 50),
            focal_gamma=config.get("focal_gamma", 2.0),
            excitation_weight=config.get("excitation_weight", 0.3),
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        model = HawkesTopicModel(
            K=K,
            beta=beta,
            theta=theta,
            times=times,
            citation_weight=config["citation_weight"],
            weibull_k=config["weibull_k"],
            weibull_lam=config["weibull_lam"],
            context_window=config["context_window"],
            decay_rate=config.get("decay_rate", 1.0),
        )
    model.get_topic_words(vocab, top_n=8)

    best_loss    = float("inf")
    loss_history = []

    epoch_bar = tqdm(range(1, config["epochs"] + 1),
                     desc="Epochs", unit="ep")

    for epoch in epoch_bar:
        np.random.shuffle(train_indices)
        epoch_loss  = epoch_hawkes = epoch_cit = 0.0
        n_batches = 0
        bs = config["batch_size"]

        batch_bar = tqdm(
            range(0, len(train_indices), bs),
            desc=f"  E{epoch:02d} batches", leave=False, unit="batch"
        )
        for start in batch_bar:
            batch = train_indices[start: start + bs]
            if use_torch:
                optimizer.zero_grad(set_to_none=True)
                loss, hawkes_ll, cit_loss = model.hawkes_log_likelihood(
                    batch, cit_mat
                )
                if not torch.isfinite(loss):
                    log.error("Non-finite loss detected; aborting training.")
                    raise ValueError("Non-finite loss detected in torch model")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"],
                )
                optimizer.step()
                if hasattr(model, "clamp_parameters"):
                    model.clamp_parameters()
                loss_value = float(loss.detach().cpu().item())
            else:
                loss, hawkes_ll, cit_loss = model.hawkes_log_likelihood(
                    batch, cit_mat
                )
                model.update_params(
                    lr=config["lr"],
                    grad_clip=config["grad_clip"],
                )
                loss_value = float(loss)

            epoch_loss   += loss_value
            epoch_hawkes += hawkes_ll
            epoch_cit    += cit_loss
            n_batches    += 1
            batch_bar.set_postfix(loss=f"{loss_value:.1f}")

        avg_loss   = epoch_loss   / max(n_batches, 1)
        avg_hawkes = epoch_hawkes / max(n_batches, 1)
        avg_cit    = epoch_cit    / max(n_batches, 1)
        loss_history.append(avg_loss)

        if use_torch:
            k_val = model.kernel_k
            lam_val = model.kernel_lam
            mu_mean = model.mu_mean
        else:
            k_val = model.kernel.k
            lam_val = model.kernel.lam
            mu_mean = model.mu.mean()

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.2f}",
            k=f"{k_val:.3f}",
            lam=f"{lam_val:.3f}",
        )
        log.info(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Loss {avg_loss:8.2f} | "
            f"Hawkes {avg_hawkes:8.2f} | "
            f"Cit {avg_cit:.4f} | "
            f"k={k_val:.3f} lam={lam_val:.3f} "
            f"μ_mean={mu_mean:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(config["model_save_path"])

    log.info(f"\nTraining complete. Best loss: {best_loss:.2f}")
    np.save(OUTPUT_DIR / "loss_history.npy", np.array(loss_history))

    return model, data, train_idx, test_idx

def _jsd(p, q, eps=1e-10):
    """Jensen-Shannon Divergence (0 = identical, 1 = maximally different)."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum();  q /= q.sum()
    return float(jensenshannon(p, q))


def _cosine(a, b, eps=1e-10):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def compute_perplexity(model, data, test_idx, config):
    from topic_model.lda_trainer import TextPreprocessor

    beta   = data["beta"]        # [K, V]
    vocab  = data["vocab"]       # list of V words
    times  = data["times"]
    papers = data["papers"]

    word2id = {w: i for i, w in enumerate(vocab)}
    preprocessor = TextPreprocessor()

    total_log_likelihood = 0.0
    total_word_count     = 0

    for d in tqdm(test_idx, desc="  perplexity", unit="paper"):
        t_d  = times[d]
        text = papers.iloc[d]["text"] if hasattr(papers, "iloc") else papers[d].get("text", "")
        if not text:
            continue

        # Actual word counts for this document
        tokens = preprocessor.preprocess(str(text))
        word_counts = {}
        for tok in tokens:
            wid = word2id.get(tok)
            if wid is not None:
                word_counts[wid] = word_counts.get(wid, 0) + 1

        if not word_counts:
            continue

        _, pi_k = model.predict_topic_intensity(
            t_d, context_window=config["context_window"]
        )
        pred_word_probs = pi_k @ beta                             # [V]
        pred_word_probs = pred_word_probs / (pred_word_probs.sum() + 1e-10)

        # Log-likelihood for this document
        doc_log_lik = 0.0
        doc_words   = 0
        for wid, count in word_counts.items():
            doc_log_lik += count * np.log(pred_word_probs[wid] + 1e-10)
            doc_words   += count

        total_log_likelihood += doc_log_lik
        total_word_count     += doc_words

    if total_word_count == 0:
        log.warning("Perplexity: no words found in test documents.")
        return {}

    perplexity = float(np.exp(-total_log_likelihood / total_word_count))
    log.info(f"  Perplexity (↓ better) : {perplexity:.2f}")
    return {"perplexity": perplexity}


def evaluate(model, data, test_idx, config):
    log.info("\n" + "=" * 60)
    log.info("STAGE 3: Evaluation on test papers")
    log.info("=" * 60)

    theta   = data["theta"]
    times   = data["times"]
    cit_mat = data["citation_matrix"]
    K_vals  = config["recall_at_k"]

    jsd_scores    = []
    cosine_scores = []
    recall_scores = {k: [] for k in K_vals}
    mrr_scores    = []
    auc_scores    = []

    n_no_cite = 0

    for d in tqdm(test_idx, desc="  evaluating", unit="paper"):
        t_d     = times[d]
        theta_d = theta[d]   
        hist_mask = times < t_d
        hist_idx  = np.where(hist_mask)[0]

        if len(hist_idx) == 0:
            continue

        
        lambda_k, pi_k = model.predict_topic_intensity(
            t_d, context_window=config["context_window"]
        )

        # Topic metrics
        jsd_scores.append(_jsd(pi_k, theta_d))
        cosine_scores.append(_cosine(pi_k, theta_d))

        
        cited_set = cit_mat.get(d, set())
        if not cited_set:
            n_no_cite += 1
            continue

        theta_hist = theta[hist_idx]
        dt_hist = t_d - times[hist_idx]

        if hasattr(model, "citation_scores"):
            ranking_scores = model.citation_scores(
                theta_d, theta_hist, dt=dt_hist
            )
        elif hasattr(model, "attention_weights"):
            ranking_scores = model.attention_weights(theta_d, theta_hist)
        else:
            attn_weights, _ = model.attention.forward(
                theta_d, theta_hist, return_cache=False
            )
            ranking_scores = attn_weights

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

        ranked = np.argsort(ranking_scores)[::-1]

        for k in K_vals:
            top_k_positions = set(ranked[:k])
            hits = len(cited_positions & top_k_positions)
            recall_scores[k].append(hits / len(cited_positions))

        for rank, pos in enumerate(ranked, start=1):
            if pos in cited_positions:
                mrr_scores.append(1.0 / rank)
                break

        if relevance.sum() > 0 and relevance.sum() < len(relevance):
            try:
                auc = roc_auc_score(relevance, ranking_scores)
                auc_scores.append(auc)
            except ValueError:
                pass

    n_topic  = len(jsd_scores)
    n_cite   = len(mrr_scores)

    metrics = {}

    if n_topic > 0:
        metrics["topic_JSD_mean"]    = float(np.mean(jsd_scores))
        metrics["topic_JSD_std"]     = float(np.std(jsd_scores))
        metrics["topic_Cosine_mean"] = float(np.mean(cosine_scores))
        metrics["topic_Cosine_std"]  = float(np.std(cosine_scores))

    if n_cite > 0:
        for k in K_vals:
            metrics[f"Recall@{k}"] = float(np.mean(recall_scores[k]))
        metrics["MRR"]            = float(np.mean(mrr_scores))
        if auc_scores:
            metrics["AUC-ROC"]    = float(np.mean(auc_scores))

    hawkes_coherence = compute_hawkes_coherence_time_bins(model, data, config)
    metrics.update(hawkes_coherence)

    pred_vs_actual = compare_predicted_vs_actual_words(model, data, config)
    metrics.update(pred_vs_actual)

    perplexity_result = compute_perplexity(model, data, test_idx, config)
    metrics.update(perplexity_result)

    
    log.info(f"\n{'─'*50}")
    log.info(f"Evaluation on {len(test_idx):,} test papers")
    log.info(f"  Papers with citations evaluated : {n_cite:,}")
    log.info(f"  Papers without citations        : {n_no_cite:,}")
    log.info(f"{'─'*50}")
    log.info("TOPIC PREDICTION")
    log.info(f"  JSD   (↓ better) : {metrics.get('topic_JSD_mean', float('nan')):.4f} "
             f"± {metrics.get('topic_JSD_std', float('nan')):.4f}")
    log.info(f"  Cosine(↑ better) : {metrics.get('topic_Cosine_mean', float('nan')):.4f} "
             f"± {metrics.get('topic_Cosine_std', float('nan')):.4f}")
    log.info("CITATION RANKING")
    for k in K_vals:
        log.info(f"  Recall@{k:<3d}(↑)   : {metrics.get(f'Recall@{k}', float('nan')):.4f}")
    log.info(f"  MRR     (↑)      : {metrics.get('MRR', float('nan')):.4f}")
    log.info(f"  AUC-ROC (↑)      : {metrics.get('AUC-ROC', float('nan')):.4f}")
    if "hawkes_coherence_mean" in metrics:
        log.info("HAWKES COHERENCE")
        log.info(f"  Metric          : {metrics.get('hawkes_coherence_metric')}")
        log.info(f"  Mean            : {metrics.get('hawkes_coherence_mean'):.4f}")
        log.info(f"  Std             : {metrics.get('hawkes_coherence_std'):.4f}")
    if "pred_actual_jsd" in metrics:
        log.info("PREDICTED VS ACTUAL WORDS")
        log.info(f"  Year            : {metrics.get('pred_actual_year')}")
        log.info(f"  Top-N           : {metrics.get('pred_actual_topn')}")
        log.info(f"  JSD             : {metrics.get('pred_actual_jsd'):.4f}")
        log.info(f"  Cosine          : {metrics.get('pred_actual_cosine'):.4f}")
        log.info(f"  Top-N overlap   : {metrics.get('pred_actual_topn_overlap'):.4f}")
    if "perplexity" in metrics:
        log.info("PERPLEXITY")
        log.info(f"  Perplexity (↓)  : {metrics.get('perplexity'):.2f}")
    log.info(f"{'─'*50}\n")

    # Save metrics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    metrics_path = OUTPUT_DIR / "eval_metrics_v3.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    return metrics


if __name__ == "__main__":
    model, data, train_idx, test_idx = train()

    metrics = evaluate(model, data, test_idx, CONFIG)

    log.info("\n" + "=" * 60)
    log.info("PREDICTION DEMO")
    log.info("=" * 60)

    T_star = float(CONFIG.get("prediction_year", data["times"].max())) + 0.5
    top_indices, word_probs, pi_k = model.predict_words(T_star, top_n=30)

    vocab = data["vocab"]
    print(f"\nPredicted top words at T*={T_star:.2f}:")
    print("  " + ", ".join(vocab[i] for i in top_indices))

    print(f"\nPredicted topic proportions:")
    for k, p in enumerate(pi_k):
        print(f"  Topic {k:2d}: {p:.4f}")
