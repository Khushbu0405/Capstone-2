import logging
import numpy as np
import torch
from collections import defaultdict

log = logging.getLogger(__name__)


def build_dsntm_data(papers, times, vocab, citation_matrix,
                     year_min, year_max, bin_width=2,
                     train_cutoff=None, device="cpu"):
    from topic_model.lda_trainer import TextPreprocessor

    dev = torch.device(device)
    V = len(vocab)
    word2id = {w: i for i, w in enumerate(vocab)}
    preprocessor = TextPreprocessor()

    # Build BoW matrix 
    D = len(papers)
    bow = np.zeros((D, V), dtype=np.float32)
    texts = papers["text"].fillna("").tolist()

    for d, text in enumerate(texts):
        for token in preprocessor.preprocess(text):
            wid = word2id.get(token)
            if wid is not None:
                bow[d, wid] += 1.0

    # Assign time bins 
    bin_edges = []
    year = year_min
    while year < year_max + 1:
        end = min(year + bin_width, year_max + 1)
        bin_edges.append((year, end))
        year = end
    T = len(bin_edges)

    time_ids = np.zeros(D, dtype=np.int64)
    for d in range(D):
        y = times[d]
        assigned = False
        for t, (ystart, yend) in enumerate(bin_edges):
            if ystart <= y < yend:
                time_ids[d] = t
                assigned = True
                break
        if not assigned:
            time_ids[d] = T - 1  # clamp to last bin

    log.info(f"DSNTM data: D={D}, V={V}, T={T} bins")
    for t, (ys, ye) in enumerate(bin_edges):
        n_docs = (time_ids == t).sum()
        log.info(f"  Bin {t}: [{ys}, {ye})  → {n_docs:,} docs")

    #RNN input: mean BoW per time bin 
    rnn_inp = np.zeros((T, V), dtype=np.float32)
    for t in range(T):
        mask = time_ids == t
        if mask.sum() > 0:
            rnn_inp[t] = bow[mask].mean(axis=0)

    #Convert to tensors 
    bow_t = torch.as_tensor(bow, dtype=torch.float32, device=dev)
    time_ids_t = torch.as_tensor(time_ids, dtype=torch.long, device=dev)
    rnn_inp_t = torch.as_tensor(rnn_inp, dtype=torch.float32, device=dev)

    result = {
        "bow": bow_t,
        "time_ids": time_ids_t,
        "rnn_inp": rnn_inp_t,
        "T": T,
        "V": V,
        "bin_edges": bin_edges,
        "doc_indices": np.arange(D),
    }

    #Train/val/test split 
    if train_cutoff is not None:
        train_end = float(train_cutoff + 1)
        train_val_mask = times < train_end
        test_mask = ~train_val_mask

        train_val_idx = np.where(train_val_mask)[0]
        test_idx_arr = np.where(test_mask)[0]

        rng = np.random.RandomState(42)
        rng.shuffle(train_val_idx)
        split = int(0.8 * len(train_val_idx))
        train_idx_arr = train_val_idx[:split]
        val_idx_arr = train_val_idx[split:]

        result["train_idx"] = train_idx_arr
        result["val_idx"] = val_idx_arr
        result["test_idx"] = test_idx_arr

        log.info(
            f"  Train: {len(train_idx_arr):,}  Val: {len(val_idx_arr):,}  "
            f"Test: {len(test_idx_arr):,}"
        )

        for name, idx_arr in [
            ("train_rnn_inp", train_idx_arr),
            ("val_rnn_inp", val_idx_arr),
        ]:
            rnn = np.zeros((T, V), dtype=np.float32)
            for t in range(T):
                mask = np.isin(np.arange(D), idx_arr) & (time_ids == t)
                if mask.sum() > 0:
                    rnn[t] = bow[mask].mean(axis=0)
            result[name] = torch.as_tensor(rnn, dtype=torch.float32, device=dev)

        for split_name, split_idx in [
            ("train_bow_by_time", train_idx_arr),
            ("val_bow_by_time", val_idx_arr),
            ("test_bow_by_time", test_idx_arr),
        ]:
            bbt = {}
            for t in range(T):
                mask = np.isin(np.arange(D), split_idx) & (time_ids == t)
                if mask.sum() > 0:
                    bbt[t] = torch.as_tensor(
                        bow[mask], dtype=torch.float32, device=dev
                    )
                else:
                    bbt[t] = torch.zeros(0, V, device=dev)
            result[split_name] = bbt

        #Citation matrices between time bins 
        if citation_matrix:
            doc2bin = {d: int(time_ids[d]) for d in range(D)}
            docs_in_bin = defaultdict(list)
            for d in range(D):
                if d in train_idx_arr or d in val_idx_arr:
                    docs_in_bin[int(time_ids[d])].append(d)

            citation_by_time = {}
            for t in range(1, T):
                cit_t = {}
                t_docs = docs_in_bin.get(t, [])
                if not t_docs:
                    continue
                for s in range(t):
                    s_docs = docs_in_bin.get(s, [])
                    if not s_docs:
                        continue
                    s_doc_set = set(s_docs)
                    mat = torch.zeros(
                        len(t_docs), len(s_docs),
                        dtype=torch.float32, device=dev
                    )
                    for i, td in enumerate(t_docs):
                        cited = citation_matrix.get(td, set())
                        for j, sd in enumerate(s_docs):
                            if sd in cited:
                                mat[i, j] = 1.0
                    cit_t[s] = mat
                if cit_t:
                    citation_by_time[t] = cit_t

            result["citation_by_time"] = citation_by_time
            result["docs_in_bin"] = dict(docs_in_bin)

    return result
