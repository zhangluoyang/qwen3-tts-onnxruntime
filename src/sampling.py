from __future__ import annotations

import numpy as np


def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _softmax_1d(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    total = np.sum(exp_scores)
    if not np.isfinite(total) or total <= 0.0:
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float32)
    return exp_scores / total


def _sample_from_probs(indices, probs, rng):
    cdf = np.cumsum(probs)
    sample = float(rng.random()) * float(cdf[-1])
    pos = int(np.searchsorted(cdf, sample, side="right"))
    if pos >= indices.shape[0]:
        pos = indices.shape[0] - 1
    return int(indices[pos])


def _top_k_indices(scores, top_k):
    vocab_size = scores.shape[0]
    if top_k is None or top_k <= 0 or top_k >= vocab_size:
        return np.arange(vocab_size, dtype=np.int64)
    k = int(top_k)
    if k == 1:
        return np.array([int(np.argmax(scores))], dtype=np.int64)
    return np.argpartition(scores, -k)[-k:].astype(np.int64, copy=False)


def _apply_top_p_to_indices(scores, indices, top_p):
    if top_p is None or top_p >= 1.0 or indices.shape[0] <= 1:
        return indices, None

    ordered = indices[np.argsort(scores[indices])[::-1]]
    probs = _softmax_1d(scores[ordered])
    cum = np.cumsum(probs)
    remove = cum > float(top_p)
    if remove.size:
        remove[1:] = remove[:-1]
        remove[0] = False
    keep = ~remove
    kept = ordered[keep]
    kept_probs = probs[keep]
    prob_sum = np.sum(kept_probs)
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        return ordered[:1], np.array([1.0], dtype=np.float32)
    return kept, kept_probs / prob_sum


def top_k_top_p_filter(logits, top_k=50, top_p=1.0):
    logits = logits.copy()
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = softmax(sorted_logits)
        cum = np.cumsum(probs)
        remove = cum > top_p
        if remove.size:
            remove[1:] = remove[:-1]
            remove[0] = False
            logits[order[remove]] = -np.inf
    return logits


def sample_token(logits, rng, do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if not do_sample:
        return int(np.argmax(logits))

    if temperature is not None and temperature > 0.0 and temperature != 1.0:
        scores = logits / float(temperature)
    else:
        scores = logits

    indices = _top_k_indices(scores, top_k)
    indices, probs = _apply_top_p_to_indices(scores, indices, top_p)
    if indices.shape[0] == 1:
        return int(indices[0])
    if probs is None:
        probs = _softmax_1d(scores[indices])
    return _sample_from_probs(indices, probs, rng)


def apply_repetition_penalty(logits, generated, penalty):
    if not generated or penalty is None or penalty == 1.0:
        return logits
    logits = logits.copy()
    tokens = np.unique(np.asarray(generated, dtype=np.int64))
    tokens = tokens[(tokens >= 0) & (tokens < logits.shape[-1])]
    if tokens.size == 0:
        return logits
    selected = logits[tokens]
    logits[tokens] = np.where(selected < 0, selected * penalty, selected / penalty)
    return logits