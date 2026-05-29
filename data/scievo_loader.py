import re
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import load_dataset, enable_progress_bar
from tqdm import tqdm

enable_progress_bar()  

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATASET_NAME = "Ahren09/SciEvo"

def normalise_arxiv_id(raw_id: str) -> str:
    if not raw_id or not isinstance(raw_id, str):
        return ""
    s = raw_id.strip()
    # strip URL prefix
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # strip version suffix  
    s = re.sub(r"v\d+$", "", s)
    return s.strip()

def load_scievo_raw():
    log.info(f"Loading {DATASET_NAME} from HuggingFace …")

    log.info("  [1/3] arxiv subset …")
    ds_arxiv = load_dataset(DATASET_NAME, "arxiv", split="train")
    papers_arxiv = pd.DataFrame(
        tqdm(ds_arxiv, desc="  arxiv → DataFrame", total=len(ds_arxiv), unit="row")
    )
    log.info(f"        {len(papers_arxiv):,} rows loaded")

    log.info("  [2/3] references subset …")
    ds_refs = load_dataset(DATASET_NAME, "references", split="train")
    refs_df = pd.DataFrame(
        tqdm(ds_refs, desc="  refs  → DataFrame", total=len(ds_refs), unit="row")
    )
    log.info(f"        {len(refs_df):,} rows loaded")

    log.info("  [3/3] semantic_scholar subset …")
    ds_s2 = load_dataset(DATASET_NAME, "semantic_scholar", split="train")
    s2_df = pd.DataFrame(
        tqdm(ds_s2, desc="  s2    → DataFrame", total=len(ds_s2), unit="row")
    )
    log.info(f"        {len(s2_df):,} rows loaded")

    return papers_arxiv, refs_df, s2_df


def parse_continuous_time(date_str) -> float:
    if pd.isna(date_str) or str(date_str).strip() == "":
        return np.nan

    s = str(date_str).strip()

    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ):
        try:
            dt = datetime.strptime(s[: len(fmt)], fmt)
            doy = dt.timetuple().tm_yday
            return dt.year + (doy - 1) / 365.0
        except ValueError:
            pass

    m = re.search(r"(\d{4})", s)
    if m:
        return float(m.group(1))
    return np.nan

def build_paper_df(papers_arxiv: pd.DataFrame) -> pd.DataFrame:
    log.info("Building unified paper DataFrame …")

    df = papers_arxiv.copy()

    rename_map = {}
    if "id"        in df.columns: rename_map["id"]        = "paper_id"
    if "summary"   in df.columns: rename_map["summary"]   = "abstract"
    if "published" in df.columns: rename_map["published"] = "pub_date"
    df = df.rename(columns=rename_map)

    # Normalise arXiv IDs
    tqdm.pandas(desc="  normalising arXiv IDs")
    df["paper_id"] = df["paper_id"].progress_apply(normalise_arxiv_id)
    df = df[df["paper_id"] != ""].dropna(subset=["paper_id"])

    # Parse continuous time
    tqdm.pandas(desc="  parsing timestamps ")
    df["time"] = df["pub_date"].progress_apply(parse_continuous_time)
    df["year"] = df["time"].apply(lambda t: int(t) if not np.isnan(t) else np.nan)

    # Concatenated text for LDA
    df["text"] = (
        df["title"].fillna("") + " " + df["abstract"].fillna("")
    ).str.strip()

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    log.info(f"  Total papers: {len(df):,}")
    log.info(f"  Time range:   {df['time'].min():.2f} – {df['time'].max():.2f}")
    log.info(f"  Valid time:   {df['time'].notna().sum():,}")
    return df


def build_s2_to_arxiv_map(s2_df: pd.DataFrame) -> dict:
    log.info("Building S2-paperId → arXivId map …")

    s2_map = {}   # paperId → arxiv_id
    mapped = 0

    pbar = tqdm(s2_df.iterrows(), total=len(s2_df),
                desc="  mapping S2 IDs", leave=True)
    for n, (_, row) in enumerate(pbar):
        paper_id = str(row.get("paperId", "")).strip()
        if not paper_id:
            continue

        arxiv_id = ""

       
        direct = row.get("arXivId", "") or row.get("arxivId", "")
        if direct and str(direct).strip() not in ("", "None", "nan"):
            arxiv_id = normalise_arxiv_id(str(direct))

        
        if not arxiv_id:
            ext = row.get("externalIds", "")
            if isinstance(ext, str) and ext.strip():
                try:
                    ext_dict = json.loads(ext)
                    arx = ext_dict.get("ArXiv") or ext_dict.get("arxiv")
                    if arx:
                        arxiv_id = normalise_arxiv_id(str(arx))
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(ext, dict):
                arx = ext.get("ArXiv") or ext.get("arxiv")
                if arx:
                    arxiv_id = normalise_arxiv_id(str(arx))

        if paper_id and arxiv_id:
            s2_map[paper_id] = arxiv_id
            mapped += 1

        if n % 500 == 0:
            pbar.set_postfix(mapped=f"{mapped:,}")

    pbar.set_postfix(mapped=f"{mapped:,}")
    log.info(f"  S2 map entries: {len(s2_map):,}")
    return s2_map


def _parse_references_field(refs_raw):
    if refs_raw is None or (isinstance(refs_raw, float) and np.isnan(refs_raw)):
        return []

    if isinstance(refs_raw, str):
        try:
            refs_raw = json.loads(refs_raw)
        except (json.JSONDecodeError, TypeError):
            return []

    if not isinstance(refs_raw, list):
        return []

    s2_ids = []
    for item in refs_raw:
        if not isinstance(item, dict):
            continue
        cited = item.get("citedPaper", {})
        if not isinstance(cited, dict):
            continue
        pid = cited.get("paperId", "")
        if pid and isinstance(pid, str) and pid.strip():
            s2_ids.append(pid.strip())

    return s2_ids


def build_citation_dict(refs_df: pd.DataFrame,
                        s2_to_arxiv: dict,
                        valid_arxiv_ids: set) -> tuple:
    log.info("Building citation dictionary …")

    citation_dict  = {}
    citation_pairs = []
    n_citing = 0   

    pbar = tqdm(refs_df.iterrows(), total=len(refs_df),
                desc="  resolving citations", leave=True)
    for n, (_, row) in enumerate(pbar):
        raw_citing = row.get("arXivId", "")
        citing_id  = normalise_arxiv_id(str(raw_citing)) if raw_citing else ""
        if not citing_id or citing_id not in valid_arxiv_ids:
            continue

        # Parse cited papers
        s2_ids = _parse_references_field(row.get("references"))

        valid_cited = []
        for s2_id in s2_ids:
            arxiv_cited = s2_to_arxiv.get(s2_id, "")
            if arxiv_cited and arxiv_cited in valid_arxiv_ids:
                valid_cited.append(arxiv_cited)

        if valid_cited:
            citation_dict[citing_id] = valid_cited
            for cited_id in valid_cited:
                citation_pairs.append((citing_id, cited_id))
            n_citing += 1

        if n % 500 == 0:
            pbar.set_postfix(
                citing=f"{n_citing:,}",
                pairs=f"{len(citation_pairs):,}",
            )

    pbar.set_postfix(
        citing=f"{n_citing:,}",
        pairs=f"{len(citation_pairs):,}",
    )
    log.info(f"  Papers with citations: {len(citation_dict):,}")
    log.info(f"  Total citation pairs:  {len(citation_pairs):,}")
    return citation_dict, citation_pairs


def filter_papers(df: pd.DataFrame,
                  min_abstract_len=50,
                  year_min=2000,
                  year_max=2024,
                  subject_tags=None,
                  keyword_terms=None) -> pd.DataFrame:
    tags = set(subject_tags) if subject_tags else set()
    terms = [t.lower() for t in (keyword_terms or []) if isinstance(t, str) and t.strip()]
    log.info(
        f"Filtering papers (year {year_min}–{year_max}, "
        f"min abstract {min_abstract_len} chars"
        + (f", tags: {sorted(tags)}" if tags else ", no tag filter")
        + (f", keywords: {terms}" if terms else "")
        + ") …"
    )
    n0 = len(df)

    if tags:
        def _has_tag(paper_tags):
            if not isinstance(paper_tags, list):
                return False
            return any(t in tags for t in paper_tags)
        df = df[df["tags"].apply(_has_tag)]
        log.info(f"  After subject tag filter             : {len(df):,}  "
                 f"(removed {n0 - len(df):,})")

    df = df.dropna(subset=["time", "abstract"])
    log.info(f"  After dropping missing time/abstract : {len(df):,}  "
             f"(removed {n0 - len(df):,})")

    n1 = len(df)
    df = df[df["abstract"].str.len() >= min_abstract_len]
    log.info(f"  After min abstract length filter     : {len(df):,}  "
             f"(removed {n1 - len(df):,})")

    n2 = len(df)
    df = df[df["year"].between(year_min, year_max)]
    log.info(f"  After year range filter              : {len(df):,}  "
             f"(removed {n2 - len(df):,})")

    if terms:
        n3 = len(df)
        text_lower = df["text"].fillna("").str.lower()
        df = df[text_lower.apply(lambda s: any(t in s for t in terms))]
        log.info(f"  After keyword filter                : {len(df):,}  "
                 f"(removed {n3 - len(df):,})")

    df = df.reset_index(drop=True)
    log.info(f"  ── Total removed: {n0 - len(df):,}  │  Final corpus: {len(df):,} papers")
    return df


def assign_indices(df: pd.DataFrame):
    df = df.reset_index(drop=True)
    df["doc_idx"] = df.index
    id_to_idx = dict(zip(df["paper_id"], df["doc_idx"]))
    return df, id_to_idx


def build_citation_matrix(citation_pairs, id_to_idx, n_docs):
    log.info("Building citation matrix …")
    matrix = {i: set() for i in range(n_docs)}
    kept = 0
    dropped_temporal = 0

    pbar = tqdm(citation_pairs, desc="  building matrix", leave=True)
    for n, (citing_id, cited_id) in enumerate(pbar):
        i = id_to_idx.get(citing_id)
        j = id_to_idx.get(cited_id)
        if i is not None and j is not None:
            if i > j:
                matrix[i].add(j)
                kept += 1
            else:
                dropped_temporal += 1

        if n % 500 == 0:
            pbar.set_postfix(kept=f"{kept:,}", dropped_time=f"{dropped_temporal:,}")

    pbar.set_postfix(kept=f"{kept:,}", dropped_time=f"{dropped_temporal:,}")
    log.info(f"  Citation links (i→j, i>j): {kept:,}  "
             f"(dropped non-causal: {dropped_temporal:,})")
    return matrix

def load_scievo(year_min=2000, year_max=2024, min_abstract_len=50,
                subject_tags=None, keyword_terms=None):
    # Step 1/6: Load raw subsets
    log.info("Step 1/6: Loading raw SciEvo subsets from HuggingFace …")
    papers_arxiv, refs_df, s2_df = load_scievo_raw()

    # Step 2/6: Build paper table
    log.info("Step 2/6: Building unified paper DataFrame …")
    papers = build_paper_df(papers_arxiv)

    # Step 3/6: Filter by tags / year / abstract length
    log.info("Step 3/6: Filtering papers …")
    papers = filter_papers(papers,
                           min_abstract_len=min_abstract_len,
                           year_min=year_min,
                           year_max=year_max,
                           subject_tags=subject_tags,
                           keyword_terms=keyword_terms)

    # Step 4/6: Assign integer indices
    log.info("Step 4/6: Assigning document indices …")
    papers, id_to_idx = assign_indices(papers)

    # Step 5/6: Build S2 - arXiv lookup then resolve citations
    log.info("Step 5/6: Building S2 paperId → arXivId bridge …")
    s2_to_arxiv = build_s2_to_arxiv_map(s2_df)

    log.info("Step 6/6: Resolving citations and building citation matrix …")
    valid_ids = set(papers["paper_id"].tolist())
    citation_dict, citation_pairs = build_citation_dict(
        refs_df, s2_to_arxiv, valid_ids
    )
    citation_matrix = build_citation_matrix(citation_pairs, id_to_idx, len(papers))

    log.info("\n=== Dataset Summary ===")
    log.info(f"  Total papers:          {len(papers):,}")
    log.info(f"  Time range:            {papers['time'].min():.3f} – "
             f"{papers['time'].max():.3f}")
    log.info(f"  Papers with citations: "
             f"{sum(1 for v in citation_matrix.values() if v):,}")

    return papers, id_to_idx, citation_dict, citation_matrix


if __name__ == "__main__":
    papers, id_to_idx, citation_dict, citation_matrix = load_scievo(
        year_min=2015, year_max=2023
    )

    print("\nSample papers:")
    print(papers[["paper_id", "title", "year", "time"]].head(5).to_string())

    print(f"\nSample citations:")
    sample_id = papers["paper_id"].iloc[100]
    if sample_id in citation_dict:
        print(f"  {sample_id} cites: {citation_dict[sample_id][:3]}")
    else:
        print("  No citations found for sample paper")