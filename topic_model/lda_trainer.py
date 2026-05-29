import re
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

class TextPreprocessor:
    """Clean and tokenise text for LDA."""

    def __init__(self, extra_stopwords=None):
        self.stop_words = set(stopwords.words("english"))

        domain_stops = {
            "paper", "propose", "proposed", "method", "approach",
            "result", "show", "shown", "use", "used", "using",
            "based", "model", "models", "also", "however",
            "two", "one", "three", "new", "work", "works",
            "study", "studies", "present", "presented", "introduction",
            "conclusion", "abstract", "experiment", "experiments",
            "dataset", "data", "evaluation", "performance", "state",
            "art", "existing", "previous", "recent", "novel",
            "task", "tasks", "baseline", "baselines", "problem", "problems",
        }
        self.stop_words.update(domain_stops)
        if extra_stopwords:
            self.stop_words.update(extra_stopwords)

        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return []

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+",        " ", text)
        text = re.sub(r"\d+",            " ", text)
        text = re.sub(r"[^a-z\s]",       " ", text)
        text = re.sub(r"\s+",            " ", text).strip()

        tokens = [
            t for t in text.split()
            if t not in self.stop_words and len(t) > 2
        ]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
        return tokens

    def preprocess_corpus(self, texts):
        """texts : list[str] → list[list[str]]"""
        log.info(f"Preprocessing {len(texts):,} documents …")
        tokenized = [
            self.preprocess(t)
            for t in tqdm(texts, desc="  tokenising", unit="doc")
        ]
        lengths = [len(t) for t in tokenized]
        log.info(f"  Avg tokens/doc : {np.mean(lengths):.1f}")
        log.info(f"  Empty docs     : {sum(1 for l in lengths if l == 0):,}")
        return tokenized

def build_gensim_corpus(tokenized_docs,
                        no_below=10, no_above=0.7, keep_n=10000):
    log.info("Building gensim dictionary …")
    dictionary = corpora.Dictionary(tokenized_docs)
    log.info(f"  Raw vocab size:      {len(dictionary):,}")

    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    log.info(f"  Filtered vocab size: {len(dictionary):,}")

    log.info("  Building BoW corpus …")
    bow_corpus = [
        dictionary.doc2bow(doc)
        for doc in tqdm(tokenized_docs, desc="  doc2bow", unit="doc")
    ]
    vocab = [dictionary[i] for i in range(len(dictionary))]
    return dictionary, bow_corpus, vocab

def train_lda(bow_corpus, dictionary, K=20,
              passes=20, iterations=400,
              alpha="auto", eta="auto", random_state=42):
    log.info(f"Training LDA: K={K}, passes={passes} …")

    lda = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=K,
        passes=passes,
        iterations=iterations,
        alpha=alpha,
        eta=eta,
        random_state=random_state,
        eval_every=None,
        minimum_probability=0.0,
        callbacks=None,       # gensim callbacks are unreliable across versions
    )
    log.info("LDA training complete.")
    return lda


def extract_beta(lda_model, vocab_size):
    log.info("Extracting β (word-topic distributions) …")
    K = lda_model.num_topics
    beta = np.zeros((K, vocab_size))

    for k in tqdm(range(K), desc="  β rows"):
        for word_id, prob in lda_model.get_topic_terms(k, topn=vocab_size):
            if word_id < vocab_size:
                beta[k, word_id] = prob

    row_sums = beta.sum(axis=1, keepdims=True)
    beta /= (row_sums + 1e-10)
    log.info(f"  β shape: {beta.shape}")
    return beta


def extract_theta(lda_model, bow_corpus, n_docs):
    log.info(f"Extracting θ for {n_docs:,} documents …")
    K = lda_model.num_topics
    theta = np.zeros((n_docs, K))

    for d, bow in enumerate(tqdm(bow_corpus, desc="  θ rows", unit="doc")):
        if not bow:
            theta[d] = np.ones(K) / K
            continue
        for topic_id, prob in lda_model.get_document_topics(
                bow, minimum_probability=0.0):
            theta[d, topic_id] = prob

    row_sums = theta.sum(axis=1, keepdims=True)
    theta /= (row_sums + 1e-10)
    log.info(f"  θ shape: {theta.shape}")
    log.info(f"  avg dominant topic prob: {theta.max(axis=1).mean():.3f}")
    return theta

def evaluate_topics(lda_model, tokenized_docs, dictionary, n_top_words=10):
    log.info("Evaluating topic coherence (NPMI) …")
    cm = CoherenceModel(
        model=lda_model, texts=tokenized_docs,
        dictionary=dictionary, coherence="c_npmi", topn=n_top_words,
    )
    score = cm.get_coherence()
    log.info(f"  NPMI Coherence: {score:.4f}")

    print("\n=== Top Words Per Topic ===")
    for k in range(lda_model.num_topics):
        words_str = ", ".join(w for w, _ in lda_model.show_topic(k, topn=10))
        print(f"  Topic {k:2d}: {words_str}")

    return score


def compute_topic_diversity(beta, n_top_words=25):
    K = beta.shape[0]
    unique_words = set()
    for k in range(K):
        unique_words.update(beta[k].argsort()[-n_top_words:].tolist())
    diversity = len(unique_words) / (K * n_top_words)
    log.info(f"  Topic Diversity: {diversity:.4f}")
    return diversity


def save_lda_outputs(beta, theta, vocab, lda_model, save_dir="./lda_outputs"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "beta.npy",  beta)
    np.save(save_dir / "theta.npy", theta)
    with open(save_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    lda_model.save(str(save_dir / "lda_model"))
    log.info(f"LDA outputs saved to {save_dir}/")


def load_lda_outputs(save_dir="./lda_outputs"):
    save_dir = Path(save_dir)
    beta  = np.load(save_dir / "beta.npy")
    theta = np.load(save_dir / "theta.npy")
    with open(save_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    lda_model = LdaModel.load(str(save_dir / "lda_model"))
    log.info(f"LDA loaded from {save_dir}/  "
             f"β:{beta.shape}  θ:{theta.shape}  V:{len(vocab)}")
    return beta, theta, vocab, lda_model


def run_lda_pipeline(papers_df, K=20, passes=20,
                     no_below=10, no_above=0.7, keep_n=10000,
                     use_bigrams=True, bigram_min_count=20, bigram_threshold=10.0,
                     save_dir="./lda_outputs", load_if_exists=True):
    save_path = Path(save_dir)
    if load_if_exists and (save_path / "beta.npy").exists():
        log.info("Found cached LDA outputs → loading …")
        return load_lda_outputs(save_dir)

    texts = papers_df["text"].fillna("").tolist()
    D = len(texts)

    preprocessor = TextPreprocessor()
    tokenized = preprocessor.preprocess_corpus(texts)

    if use_bigrams:
        log.info("Learning bigrams …")
        phrases = Phrases(tokenized, min_count=bigram_min_count, threshold=bigram_threshold)
        bigram = Phraser(phrases)
        tokenized = [bigram[doc] for doc in tokenized]

    log.info(f"  Non-empty docs: {sum(len(t)>0 for t in tokenized):,} / {D:,}")

    dictionary, bow_corpus, vocab = build_gensim_corpus(
        tokenized, no_below=no_below, no_above=no_above, keep_n=keep_n
    )
    V = len(vocab)

    lda_model = train_lda(bow_corpus, dictionary, K=K, passes=passes)

    beta  = extract_beta(lda_model, V)
    theta = extract_theta(lda_model, bow_corpus, D)

    evaluate_topics(lda_model, tokenized, dictionary)
    compute_topic_diversity(beta)

    save_lda_outputs(beta, theta, vocab, lda_model, save_dir)

    return beta, theta, vocab, lda_model

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.scievo_loader import load_scievo

    papers, id_to_idx, citation_dict, citation_matrix = load_scievo(
        year_min=2018, year_max=2023
    )
    beta, theta, vocab, lda_model = run_lda_pipeline(
        papers, K=20, passes=10, save_dir="./lda_outputs_test"
    )
    print(f"\nβ: {beta.shape}  θ: {theta.shape}  V: {len(vocab)}")
    print(f"\nSample θ (first 3 docs):\n{theta[:3].round(3)}")