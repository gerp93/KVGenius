"""
Semantic retrieval engine for KVGenius.

Provides embedding-based similarity search for memories (now) and lorebook
entries (future). Uses a lightweight sentence-transformer model that runs on
CPU so it never competes with the chat model for VRAM.

NOTE: The custom PyTorch build (sm_120 / Blackwell) has a bug in its default
SDPA implementation that produces NaN or degenerate embeddings. We work around
this by forcing the *MATH* SDP backend and encoding texts one-at-a-time to
avoid padding-related contamination. This is slightly slower than batch
encoding but produces correct cosine-similarity scores.

Usage:
    from core.semantic_index import get_retriever

    retriever = get_retriever()
    results = retriever.query("What happened at the cave?", texts, top_k=5)
    # results: [ScoredChunk(text="...", score=0.82, index=3), ...]
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via settings.yaml)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # ~80 MB, very fast on CPU
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.25  # Minimum cosine similarity to include
DEFAULT_MAX_TOKENS = 400  # Rough token budget for injected memories
AVG_CHARS_PER_TOKEN = 4   # Rough estimate for token counting

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoredChunk:
    """A text chunk with its similarity score and original index."""
    text: str
    score: float
    index: int
    source: str = ""  # e.g. "auto", "manual", "lorebook"
    pinned: bool = False

    def __repr__(self):
        return f"ScoredChunk(score={self.score:.3f}, text={self.text[:60]!r})"


@dataclass
class RetrievalResult:
    """Full result set from a retrieval query."""
    query: str
    selected: List[ScoredChunk]       # Chunks that made the cut
    rejected: List[ScoredChunk]       # Chunks that were below threshold
    total_available: int              # How many chunks were in the pool
    budget_tokens_used: int = 0       # Estimated tokens consumed
    budget_tokens_max: int = 0        # Token budget that was applied

    @property
    def selected_texts(self) -> List[str]:
        return [c.text for c in self.selected]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """
    Embeds text chunks and queries, returns top-K by cosine similarity.

    The embedding model is loaded lazily on first use and cached for the
    lifetime of the process. It runs on CPU to avoid VRAM contention.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._tokenizer = None   # Lazy-loaded
        self._transformer = None  # Lazy-loaded

    # -- Public API ----------------------------------------------------------

    def query(
        self,
        query_text: str,
        chunks: Sequence[str],
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        max_token_budget: int = DEFAULT_MAX_TOKENS,
        sources: Optional[Sequence[str]] = None,
        pinned_indices: Optional[Sequence[int]] = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query_text:      The user's message (or any query string).
            chunks:          All candidate text chunks (memories, lore entries, etc.).
            top_k:           Maximum number of chunks to return.
            min_score:       Minimum cosine similarity threshold.
            max_token_budget: Approximate token budget for selected chunks.
            sources:         Optional parallel list of source labels per chunk.
            pinned_indices:  Indices of chunks that MUST be included regardless
                             of score (e.g. user-pinned memories).

        Returns:
            RetrievalResult with selected and rejected chunks.
        """
        if not chunks:
            return RetrievalResult(
                query=query_text, selected=[], rejected=[],
                total_available=0, budget_tokens_max=max_token_budget,
            )

        pinned_set = set(pinned_indices or [])
        src_labels = sources or [""] * len(chunks)

        # Embed everything (one-at-a-time to avoid padding/SDPA bug)
        all_texts = [query_text] + list(chunks)
        embeddings = self._encode_texts(all_texts)

        query_emb = embeddings[0]   # (dim,)
        chunk_embs = embeddings[1:]  # (N, dim)

        # Cosine similarity (embeddings are already normalized)
        scores = np.dot(chunk_embs, query_emb)  # (N,)

        # Build scored list
        scored: List[ScoredChunk] = []
        for i, (text, score) in enumerate(zip(chunks, scores)):
            scored.append(ScoredChunk(
                text=text,
                score=float(score),
                index=i,
                source=src_labels[i] if i < len(src_labels) else "",
                pinned=i in pinned_set,
            ))

        # Sort by: pinned first, then by score descending
        scored.sort(key=lambda c: (not c.pinned, -c.score))

        # Select within budget
        selected: List[ScoredChunk] = []
        rejected: List[ScoredChunk] = []
        token_budget_remaining = max_token_budget

        for chunk in scored:
            # Pinned chunks always included
            if chunk.pinned:
                est_tokens = max(1, len(chunk.text) // AVG_CHARS_PER_TOKEN)
                selected.append(chunk)
                token_budget_remaining -= est_tokens
                continue

            # Score threshold
            if chunk.score < min_score:
                rejected.append(chunk)
                continue

            # Top-K limit
            if len(selected) >= top_k:
                rejected.append(chunk)
                continue

            # Token budget
            est_tokens = max(1, len(chunk.text) // AVG_CHARS_PER_TOKEN)
            if est_tokens > token_budget_remaining:
                rejected.append(chunk)
                continue

            selected.append(chunk)
            token_budget_remaining -= est_tokens

        budget_used = max_token_budget - token_budget_remaining

        return RetrievalResult(
            query=query_text,
            selected=selected,
            rejected=rejected,
            total_available=len(chunks),
            budget_tokens_used=budget_used,
            budget_tokens_max=max_token_budget,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return normalized embeddings for a list of texts.
        Useful for pre-computing or caching embeddings.
        """
        return self._encode_texts(list(texts))

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None

    def unload(self):
        """Free the embedding model from memory."""
        if self._tokenizer is not None:
            del self._tokenizer
            del self._transformer
            self._tokenizer = None
            self._transformer = None
            logger.info("Semantic retriever model unloaded")

    # -- Internal ------------------------------------------------------------

    def _load_model(self):
        """Lazy-load tokenizer + transformer (no SentenceTransformer wrapper)."""
        if self._tokenizer is None:
            logger.info(f"Loading semantic embedding model: {self._model_name}")
            import torch
            from transformers import AutoTokenizer, AutoModel

            hf_name = f"sentence-transformers/{self._model_name}"
            self._tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self._transformer = AutoModel.from_pretrained(hf_name)
            self._transformer.eval()
            logger.info(f"Semantic model loaded ({self._model_name})")

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single string → L2-normalized (dim,) numpy array."""
        import torch

        self._load_model()
        encoded = self._tokenizer(
            text, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            with torch.no_grad():
                outputs = self._transformer(**encoded)

        # Mean pooling over non-padding tokens
        mask = encoded["attention_mask"].unsqueeze(-1).expand(
            outputs.last_hidden_state.size()
        ).float()
        pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(
            mask.sum(1), min=1e-9,
        )
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normed.squeeze(0).numpy()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings → (N, dim) L2-normalized numpy array.

        Each text is encoded individually to avoid padding-related degenerate
        embeddings caused by the custom PyTorch SDPA bug.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return np.stack([self._encode_single(t) for t in texts])


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_retriever: Optional[SemanticRetriever] = None


def get_retriever(model_name: str = DEFAULT_MODEL_NAME) -> SemanticRetriever:
    """Get or create the global SemanticRetriever singleton."""
    global _retriever
    if _retriever is None or _retriever._model_name != model_name:
        _retriever = SemanticRetriever(model_name=model_name)
    return _retriever


def retrieve_memories(
    query_text: str,
    memories: List[dict],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_token_budget: int = DEFAULT_MAX_TOKENS,
) -> RetrievalResult:
    """
    Convenience function: retrieve the most relevant memories for a user message.

    Args:
        query_text:      The user's chat message.
        memories:        List of memory dicts from DB (must have 'content' and 'source' keys).
        top_k:           Max memories to inject.
        min_score:       Minimum similarity threshold.
        max_token_budget: Token budget cap.

    Returns:
        RetrievalResult with scored + filtered memories.
    """
    if not memories:
        return RetrievalResult(
            query=query_text, selected=[], rejected=[],
            total_available=0, budget_tokens_max=max_token_budget,
        )

    texts = [m["content"] for m in memories]
    sources = [m.get("source", "") for m in memories]

    # Pinned = manual memories always get included
    pinned = [i for i, m in enumerate(memories) if m.get("source") == "manual"]

    retriever = get_retriever()
    return retriever.query(
        query_text,
        texts,
        top_k=top_k,
        min_score=min_score,
        max_token_budget=max_token_budget,
        sources=sources,
        pinned_indices=pinned,
    )
