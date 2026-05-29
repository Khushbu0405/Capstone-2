import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
_TOPIC_CMAP = plt.cm.get_cmap("tab20")

def _topic_colour(k, n_topics):
    return _TOPIC_CMAP(k / max(n_topics - 1, 1))
def plot_topic_words(beta,
                     vocab,
                     top_n: int = 12,
                     ncols: int = 5,
                     bar_height: float = 0.55,
                     save_path: str = str(OUTPUT_DIR / "topic_words.png"),
                     show: bool = False):
    K = beta.shape[0]
    nrows = int(np.ceil(K / ncols))

    fig_w  = ncols * 3.4
    fig_h  = nrows * (top_n * 0.28 + 0.7)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             constrained_layout=True)
    axes_flat = axes.flatten() if K > 1 else [axes]

    for k in range(K):
        ax = axes_flat[k]
        top_ids   = beta[k].argsort()[-top_n:][::-1]
        top_probs = beta[k][top_ids]
        top_words = [vocab[i] for i in top_ids]

        # Plot bottom-to-top so highest prob is at the top of the chart
        y_pos = np.arange(top_n)
        colour = _topic_colour(k, K)
        ax.barh(y_pos, top_probs[::-1], height=bar_height,
                color=colour, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words[::-1], fontsize=8)
        ax.set_title(f"Topic {k}", fontsize=9, fontweight="bold", pad=3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for k in range(K, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("LDA Topic–Word Distributions", fontsize=13,
                 fontweight="bold", y=1.01)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] topic-word chart saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def plot_topic_heatmap(theta,
                       times,
                       topic_labels=None,
                       n_bins: int = 40,
                       time_fmt: str = "year",
                       cmap: str = "YlOrRd",
                       save_path: str = str(OUTPUT_DIR / "topic_heatmap.png"),
                       show: bool = False):
    K = theta.shape[1]
    D = theta.shape[0]

    if topic_labels is None:
        topic_labels = [f"T{k}" for k in range(K)]

    t_min, t_max = times.min(), times.max()

    # Bin edges
    bin_edges  = np.linspace(t_min, t_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Accumulate
    heat = np.zeros((K, n_bins))
    counts = np.zeros(n_bins)

    for d in range(D):
        b = np.searchsorted(bin_edges[1:], times[d])
        b = min(b, n_bins - 1)
        heat[:, b] += theta[d]
        counts[b]  += 1

    mask = counts > 0
    heat[:, mask] /= counts[mask]

    if time_fmt == "year":
        xtick_labels = [str(int(c)) for c in bin_centres]
    else:
        xtick_labels = [f"{c:.1f}" for c in bin_centres]

    
    step = max(1, n_bins // 10)
    shown_ticks   = list(range(0, n_bins, step))
    shown_labels  = [xtick_labels[i] for i in shown_ticks]

    fig_w = max(14, n_bins * 0.38)
    fig_h = max(5,  K    * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        heat,
        ax=ax,
        cmap=cmap,
        xticklabels=False,
        yticklabels=topic_labels,
        linewidths=0,
        cbar_kws={"label": "Mean θ_d[k]", "shrink": 0.7},
    )

    ax.set_xticks(shown_ticks)
    ax.set_xticklabels(shown_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(topic_labels, fontsize=8)

    ax.set_xlabel("Publication time", fontsize=10)
    ax.set_ylabel("Topic",            fontsize=10)
    ax.set_title(
        f"Topic Proportion Over Time  (D={D:,}, K={K}, bins={n_bins})",
        fontsize=11, fontweight="bold", pad=8
    )

    ax2 = ax.twinx()
    ax2.plot(np.arange(n_bins) + 0.5, counts,
             color="steelblue", alpha=0.45, linewidth=1.2, label="# papers")
    ax2.set_ylabel("# papers in bin", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=8)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.5)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] topic heatmap saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig

def visualize_lda(beta, theta, times, vocab,
                  top_n_words: int = 12,
                  n_time_bins: int = 40,
                  out_dir: str = str(OUTPUT_DIR),
                  topic_labels=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    words_path   = f"{out_dir}/topic_words.png"
    heatmap_path = f"{out_dir}/topic_heatmap.png"

    plot_topic_words(
        beta, vocab,
        top_n=top_n_words,
        save_path=words_path,
    )

    plot_topic_heatmap(
        theta, times,
        topic_labels=topic_labels,
        n_bins=n_time_bins,
        save_path=heatmap_path,
    )

    return words_path, heatmap_path

if __name__ == "__main__":
    import pickle, sys
    sys.path.append("..")

    lda_dir = str(OUTPUT_DIR / "lda")
    import numpy as np

    try:
        beta  = np.load(f"{lda_dir}/beta.npy")
        theta = np.load(f"{lda_dir}/theta.npy")
        with open(f"{lda_dir}/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        times = np.random.uniform(2015, 2023, theta.shape[0])

        w_path, h_path = visualize_lda(
            beta, theta, times, vocab,
            top_n_words=12, n_time_bins=30, out_dir=str(OUTPUT_DIR)
        )
        print(f"Topic-word chart : {w_path}")
        print(f"Topic heatmap    : {h_path}")

    except FileNotFoundError:
        print("No saved LDA outputs found. Run train.py first.")