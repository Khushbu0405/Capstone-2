## Key Idea

Scientific publishing is **self-exciting** - one influential paper triggers follow-up work, which triggers more work. We model this with a Hawkes process where each past paper's contribution to future topic intensity is:

```
λ_k(T*) = μ_k + Σ  θ_e[k] · A_{e,ctx} · f_Δ(T* − t_e)
              ↑         ↑          ↑            ↑
          base rate   topic wt   attention   Weibull decay
```

- **Frozen LDA** (K=30) provides stable topic representations (β, θ)
- **Weibull kernel** captures delayed citation peaks (unlike exponential decay)
- **Citation-supervised attention** (BCE loss) aligns learned weights with real citations
- **Two tasks:** topic prediction + citation recommendation

---

## Experiment Setup

| Component | Value |
|-----------|-------|
| Corpus | arXiv `cs.CL` + `cs.LG` (2014–2021) |
| Topics | K = 30 |
| Train | Papers up to 2020 |
| Test | Papers from 2021 |
| Source | [Ahren09/SciEvo](https://huggingface.co/datasets/Ahren09/SciEvo) (HuggingFace) |

---

## Repository Structure

```
├── train_v3.py                  # Main Hawkes training + evaluation
├── hawkes/
│   └── torch_model_v3.py        # Hawkes topic model (PyTorch)
├── topic_model/
│   └── lda_trainer.py           # LDA training and preprocessing
├── data/
│   └── scievo_loader.py         # SciEvo dataset loader
├── train_dsntm_baseline.py      # DSNTM training + eval
├── dsntm_model.py               # DSNTM model class
├── dsntm_data.py                # DSNTM data structures
├── simple_baselines.py          # Static LDA, Uniform Hawkes, Content+Recency
├── compare_models.py            # Side-by-side metric comparison
├── check_corpus.py              # Corpus diagnostics
├── visualize.py                 # Plot generation
└── outputs/                     # All generated outputs
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Hawkes model (V3)

```bash
python train_v3.py
```

**Outputs:**
- `outputs/eval_metrics_v3.json` — all evaluation metrics
- `outputs/lda_2014_2021_cscl_cslg_k30/` — LDA model + frozen β
- `outputs/processed_data_2014_2021_cscl_cslg_k30.pkl` — processed corpus
- `outputs/visualizations_v3_cscl_cslg_k30/` — plots

> First run downloads the HuggingFace dataset `Ahren09/SciEvo` and caches the processed pickle. Subsequent runs reuse it.

### 2. Run the DSNTM baseline

DSNTM defaults to K=50 and tries to load the entire BOW onto GPU, which OOMs on most cards. Pass `--K 30` and run on CPU:

```bash
CUDA_VISIBLE_DEVICES="" python train_dsntm_baseline.py \
    --data_save_path outputs/processed_data_2014_2021_cscl_cslg_k30.pkl \
    --K 30 \
    --epochs 10
```

**Outputs:** `outputs/dsntm_baseline/dsntm_baseline_metrics.json`

> If your GPU has enough free memory, drop the `CUDA_VISIBLE_DEVICES=""` prefix.

### 3. Run the simple baselines

```bash
python simple_baselines.py \
    --data_save_path outputs/processed_data_2014_2021_cscl_cslg_k30.pkl
```

### 4. Corpus diagnostics

```bash
python check_corpus.py \
    --data_save_path outputs/processed_data_2014_2021_cscl_cslg_k30.pkl \
    --train_cutoff 2020
```

Prints papers-per-year, train/test split sizes, and topic homogeneity warnings.

### 5. Compare models

```bash
python compare_models.py \
    --hawkes_metrics outputs/eval_metrics_v3.json \
    --dsntm_metrics outputs/dsntm_baseline/dsntm_baseline_metrics.json
```

---

## Data

The pipeline downloads `Ahren09/SciEvo` from HuggingFace on the first run and caches processed data under `outputs/`. All subsequent runs reuse the cached pickle — delete it if you change preprocessing or category filters.

---

## References

- He et al., *"HawkesTopic: A Joint Model for Network Inference and Topic Modeling from Text-Based Cascades"*, ICML 2015
- Miyamoto et al., *"Dynamic Structured Neural Topic Model with Self-Attention Mechanism"*, ACL Findings 2023
- Blei et al., *"Latent Dirichlet Allocation"*, JMLR 2003
- Hawkes, *"Spectra of Some Self-Exciting and Mutually Exciting Point Processes"*, Biometrika 1971
