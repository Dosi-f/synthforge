"""
Quality and diversity metrics for evaluating generated datasets.

GPU-dependent: embedding-based metrics use sentence-transformers,
which benefits significantly from GPU acceleration for large datasets.

TODO:
- [ ] Implement RewardModelScorer using ArmoRM or similar
- [ ] Add n-gram diversity metrics (faster, no GPU needed)
- [ ] Implement Self-BLEU for output diversity
- [ ] Add prompt-difficulty estimation
"""

from typing import List, Optional
import numpy as np

from synthforge.generator import GenerationSample


class DiversityAnalyzer:
    """
    Analyze semantic diversity of a dataset using embeddings.

    Uses cosine similarity to detect near-duplicate samples.
    GPU acceleration via sentence-transformers is strongly recommended
    for datasets > 1,000 samples.

    Args:
        model_name: sentence-transformers model name.
        similarity_threshold: Cosine similarity above which samples are
                              considered near-duplicates.
        device: 'cpu', 'rocm', or None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def analyze(self, samples: List[GenerationSample]) -> dict:
        """
        Compute diversity metrics for a set of samples.

        Returns:
            dict with:
            - total_samples: int
            - near_duplicate_pairs: list of (idx_a, idx_b, similarity)
            - unique_ratio: fraction of samples that are "unique"
            - mean_pairwise_similarity: float
        """
        self._load_model()

        # Extract assistant responses for embedding
        texts = []
        for s in samples:
            for msg in reversed(s.messages):
                if msg.get("role") == "assistant":
                    texts.append(msg.get("content", ""))
                    break
            else:
                texts.append("")

        if not texts:
            return {"error": "No texts to embed"}

        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=32,  # TODO: Make configurable
        )

        # Compute cosine similarity matrix
        # TODO: For very large datasets (>50K), use FAISS for efficient search
        similarities = self._cosine_similarity_matrix(embeddings)

        # Find near-duplicates (upper triangle only, exclude diagonal)
        n = len(texts)
        near_dupes = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] >= self.similarity_threshold:
                    near_dupes.append(
                        {
                            "idx_a": i,
                            "idx_b": j,
                            "similarity": float(similarities[i, j]),
                        }
                    )

        # Compute metrics
        duplicate_indices = set()
        for pair in near_dupes:
            duplicate_indices.add(pair["idx_b"])  # Second one is considered duplicate

        unique_count = n - len(duplicate_indices)

        # Mean pairwise similarity (upper triangle)
        upper_tri = similarities[np.triu_indices(n, k=1)]
        mean_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0

        return {
            "total_samples": n,
            "near_duplicate_pairs": near_dupes,
            "unique_count": unique_count,
            "unique_ratio": unique_count / n if n > 0 else 0.0,
            "mean_pairwise_similarity": mean_sim,
        }

    @staticmethod
    def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        normalized = embeddings / norms

        return normalized @ normalized.T


class QualityScorer:
    """
    Score sample quality using a reward model.

    EXPERIMENTAL — requires a separate reward model to be loaded.
    GPU strongly recommended for models > 1B parameters.

    TODO: Implement using ArmoRM or similar reward model.
    Currently a placeholder that returns dummy scores.
    """

    def __init__(self, model_name: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"):
        self.model_name = model_name
        # TODO: Load reward model
        # For now, placeholder

    def score(self, samples: List[GenerationSample]) -> List[dict]:
        """
        Score each sample.

        Returns list of {"sample_index": int, "score": float, "dimensions": dict}
        """
        # TODO: Implement actual scoring
        # Placeholder — return random scores for interface testing
        import random

        results = []
        for i in range(len(samples)):
            results.append({
                "sample_index": i,
                "score": round(random.uniform(0.3, 0.95), 3),
                "dimensions": {
                    "helpfulness": round(random.uniform(0.3, 0.95), 3),
                    "accuracy": round(random.uniform(0.3, 0.95), 3),
                    "coherence": round(random.uniform(0.3, 0.95), 3),
                },
                "note": "PLACEHOLDER — not using real reward model yet",
            })
        return results
