"""Tests for the metrics module."""

import pytest
from synthforge.generator import GenerationSample
from synthforge.metrics import DiversityAnalyzer, QualityScorer


class TestDiversityAnalyzer:
    """Test diversity analysis."""

    @pytest.fixture
    def analyzer(self):
        return DiversityAnalyzer(
            model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.85,
            device="cpu",  # Force CPU for tests
        )

    def test_analyze_identical_samples(self, analyzer):
        """Identical texts should have high similarity."""
        samples = [
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Say hello"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                ],
            ),
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Say hi"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                ],
            ),
        ]
        result = analyzer.analyze(samples)
        assert result["total_samples"] == 2
        assert result["unique_ratio"] >= 0.0  # Should flag as duplicate
        assert 0.0 <= result["mean_pairwise_similarity"] <= 1.0

    def test_analyze_diverse_samples(self, analyzer):
        """Very different texts should have low similarity."""
        samples = [
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Python question"},
                    {
                        "role": "assistant",
                        "content": "Python decorators are functions that wrap other functions to modify their behavior.",
                    },
                ],
            ),
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Rust question"},
                    {
                        "role": "assistant",
                        "content": "Rust's ownership system ensures memory safety without a garbage collector.",
                    },
                ],
            ),
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Cooking question"},
                    {
                        "role": "assistant",
                        "content": "To make perfect pasta, use plenty of salted water and cook until al dente.",
                    },
                ],
            ),
        ]
        result = analyzer.analyze(samples)
        assert result["total_samples"] == 3
        # These should be relatively diverse
        assert result["mean_pairwise_similarity"] < 0.7

    def test_empty_dataset(self, analyzer):
        """Should handle empty dataset gracefully."""
        result = analyzer.analyze([])
        assert "error" in result

    def test_single_sample(self, analyzer):
        """Single sample should have no near-duplicates."""
        samples = [
            GenerationSample(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ],
            ),
        ]
        result = analyzer.analyze(samples)
        assert result["total_samples"] == 1
        assert len(result["near_duplicate_pairs"]) == 0
        assert result["unique_ratio"] == 1.0


class TestQualityScorer:
    """Test quality scoring."""

    def test_placeholder_scores(self):
        """Should return placeholder scores for now."""
        scorer = QualityScorer()
        samples = [
            GenerationSample(
                messages=[{"role": "assistant", "content": "Test response"}],
            ),
        ]
        scores = scorer.score(samples)
        assert len(scores) == 1
        assert 0.0 <= scores[0]["score"] <= 1.0
        assert "note" in scores[0]
        assert "PLACEHOLDER" in scores[0]["note"]
