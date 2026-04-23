"""
===== Non-LLM RAG Classification Metrics =====

Metrics included:

    1. Context Precision(How much of the retrieved context is relevant)(Retrival part)

    2. Answer Exact Match (EM)(Binary exact-string match of predicted vs. true label)(not soo useful)

    3. Token-level F1(Token overlap F1 between answer and reference)

    4. ROUGE-L(Longest-common-subsequence recall/precision/F1)

    5. BM25 Retrieval Score(Sparse lexical similarity of query vs. retrieved docs)(Retrival part)

===== Excusively RAG Classification metrics =====

    6. Cohen's Kappa
Measures inter-rater agreement, adjusted for chance. Useful for comparing classifier performance to random guessing. 
Values range from -1 (no agreement) to 1 (perfect agreement).

    7. Semantic Consistency
Assesses whether the classifier's output is semantically consistent with the retrieved context. 
Uses embeddings or semantic similarity metrics (e.g., cosine similarity) to compare the classification and context

    8.RAG-Augmented _score
Compares classification _score with and without retrieved context. 
Quantifies the improvement (or degradation) due to RAG augmentation.

"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Dynamic-programming LCS length."""
    m, n = len(seq_a), len(seq_b)
    # Space-optimised DP (two rows)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]

def _parse_labels(label_str: str) -> list[str]:
    """
    Convert a comma-separated string of labels into a list.
    'python,java,python' → ['python', 'java', 'python']
    Also works for a single label: 'python' → ['python']
    """
    return [l.strip() for l in label_str.split(",") if l.strip()]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure-Python cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ===========================================================================
# Metric 1 – Context Precision
# ===========================================================================

def context_precision(
    retrieved_docs: List[str],
    relevant_doc_ids: List[int],
) -> float:
    """
    Proportion of retrieved documents that are actually relevant.

    Args:
        retrieved_docs:   Ordered list of retrieved document texts (or ids).
        relevant_doc_ids: 0-based indices of the truly relevant documents
                          within `retrieved_docs`.

    Returns:
        Precision ∈ [0, 1].

    Example:
        >>> context_precision(["d0","d1","d2","d3"], relevant_doc_ids=[0,2])
        0.5
    """
    if not retrieved_docs:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for i in range(len(retrieved_docs)) if i in relevant_set)
    return hits / len(retrieved_docs)



# ===========================================================================
# Metric 2 – Exact Match (EM)
# ===========================================================================

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Normalised exact-match: 1.0 if strings match after lowercasing and
    collapsing whitespace, else 0.0.

    Example:
        >>> exact_match("Paris", "paris")
        1.0
        >>> exact_match("Paris", "London")
        0.0
    """
    def _normalise(s: str) -> str:
        return " ".join(s.lower().split())

    return float(_normalise(prediction) == _normalise(ground_truth))


# ===========================================================================
# Metric 3 – Token-level F1
# ===========================================================================

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-overlap F1 between the predicted answer and the reference.
    Standard metric from the SQuAD benchmark.

    Returns:
        F1 score ∈ [0, 1].

    Example:
        >>> round(token_f1("the cat sat on the mat", "the cat sat"), 4)
        0.8
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ===========================================================================
# Metric 4 – ROUGE-L
# ===========================================================================

def rouge_l(prediction: str, reference: str) -> dict:
    """
    ROUGE-L based on the Longest Common Subsequence (LCS) at token level.

    Returns a dict with keys: precision, recall, f1.

    Example:
        >>> scores = rouge_l("the cat sat on the mat", "the cat sat on the wall")
        >>> round(scores["f1"], 4)
        0.9091
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        score = float(pred_tokens == ref_tokens)
        return {"precision": score, "recall": score, "f1": score}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ===========================================================================
# Metric 5 – BM25 Retrieval Score
# ===========================================================================

class BM25:
    """
    Lightweight BM25 implementation (Robertson et al.) for scoring
    how well a query matches each retrieved document.

    Parameters:
        k1: Term-frequency saturation (default 1.5).
        b:  Length normalisation (default 0.75).
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_corpus = [_tokenize(doc) for doc in corpus]
        self.N = len(self.tokenized_corpus)
        self._avgdl = (
            sum(len(d) for d in self.tokenized_corpus) / self.N if self.N else 0
        )
        self._df: Counter = Counter()
        for doc in self.tokenized_corpus:
            for term in set(doc):
                self._df[term] += 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score for a single (query, document) pair."""
        query_terms = _tokenize(query)
        doc = self.tokenized_corpus[doc_index]
        doc_len = len(doc)
        term_freq = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
            score += idf * numerator / (denominator if denominator else 1)
        return score

    def rank(self, query: str) -> List[tuple]:
        """Return (doc_index, score) pairs sorted by descending BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: x[1], reverse=True)


def bm25_retrieval_scores(query: str, corpus: List[str]) -> List[float]:
    """
    Convenience wrapper: returns the BM25 score for `query` against each
    document in `corpus`, in the original corpus order.

    Example:
        >>> docs = ["Paris is the capital of France", "Berlin is in Germany"]
        >>> scores = bm25_retrieval_scores("capital of France", docs)
        >>> scores[0] > scores[1]
        True
    """
    bm25 = BM25(corpus)
    return [bm25.score(query, i) for i in range(len(corpus))]


# ===========================================================================
# Metric 6 – Cohen's Kappa
# ===========================================================================
 
def cohen_kappa(predictions: str, ground_truths: str) -> float:
    """
    Cohen's Kappa - measures inter-rater / classifier agreement
    corrected for chance, over a corpus of label pairs.
 
        κ = (p_o - p_e) / (1 - p_e)
 
    where p_o = observed agreement, p_e = expected agreement by chance.
 
    Args:
        predictions:   Comma-separated predicted labels.
        ground_truths: Comma-separated ground-truth labels, a.k.a step (of the code)
 
    Returns:
        κ ∈ (-1, 1].  1 = perfect, 0 = chance, <0 = worse than chance.
 
    Example:
        >>> round(cohen_kappa("A,A,B,B", "A,B,B,A"), 4)
        0.0
    """
    preds  = _parse_labels(predictions)
    truths = _parse_labels(ground_truths)
 
    if len(preds) != len(truths):
        raise ValueError("Length mismatch between predictions and ground truths.")
 
    n = len(preds)
    if n == 0:
        raise ValueError("Empty label lists.")
 
    classes = sorted(set(preds) | set(truths))
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
 
    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for p, t in zip(preds, truths):
        matrix[idx[t]][idx[p]] += 1
 
    # p_o  (diagonal sum / n)
    p_observed = sum(matrix[i][i] for i in range(k)) / n
 
    # p_e  (sum over classes of: (row_i_total / n) * (col_i_total / n))
    row_totals = [sum(matrix[i]) for i in range(k)]
    col_totals = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    p_expected = sum(row_totals[i] * col_totals[i] for i in range(k)) / (n * n)
 
    if p_expected == 1.0:
        # Perfect agreement with trivial distribution
        return 1.0
 
    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return round(kappa, 4)


# ===========================================================================
# Metric 7 – Semantic Consistency
# ===========================================================================
 
def semantic_consistency(prediction: str, ground_truths: str) -> float:
    """
    Measures semantic similarity between the predicted answer and the
    ground_truths answer, capturing meaning beyond surface token overlap.
 
    Strategy (in priority order):
      1. Sentence-embedding cosine similarity  (SentenceTransformer).
      2. TF-IDF cosine similarity              (sklearn).
      3. Jaccard token overlap fallback        (always available).
 
    Args:
        prediction: Model-generated answer / rationale (str).
        ground_truths:  Ground-truth answer / rationale (str).
 
    Returns:
        Semantic consistency score ∈ [0, 1].
 
    Example:
        >>> semantic_consistency("sort in ascending order", "order ascendingly")
        0.xxxx  # model-dependent
    """
    # --- Strategy 1: Sentence-embedding cosine similarity ---
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
 
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_pred = _st_model.encode(prediction).tolist()
        emb_ref  = _st_model.encode(ground_truths).tolist()
        return round(_cosine_similarity(emb_pred, emb_ref), 4)
 
    except Exception:
        pass
 
    # --- Strategy 2: TF-IDF cosine similarity ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore
 
        corpus = [prediction, ground_truths]
        tfidf  = TfidfVectorizer().fit_transform(corpus)
        vecs   = tfidf.toarray()
        sim    = _cosine_similarity(vecs[0].tolist(), vecs[1].tolist())
        return round(sim, 4)
 
    except Exception:
        pass
 
    # --- Strategy 3: Jaccard token overlap fallback ---
    pred_set = set(_tokenize(prediction))
    ref_set  = set(_tokenize(ground_truths))
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    intersection = pred_set & ref_set
    union        = pred_set | ref_set
    return round(len(intersection) / len(union), 4)

# ===========================================================================
# Metric 8 – RAG Augmented score
# ===========================================================================

def rag_augmented_score(_with_rag: float, _without_rag: float) -> float:
    """Compute RAG-augmented _score a.k.a whatever metric"""
    return _with_rag - _without_rag




