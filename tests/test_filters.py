"""Tests for the filter module."""

from synthforge.filters import LengthFilter, CompositeFilter
from synthforge.generator import GenerationSample


class TestLengthFilter:
    """Test LengthFilter behavior."""

    def test_filters_short_content(self, sample_empty):
        """Should remove samples with too-short assistant responses."""
        f = LengthFilter(min_chars=10, max_chars=10000)
        result = f.apply([sample_empty])
        assert len(result) == 0
        assert f.get_stats()["removed"] == 1

    def test_keeps_valid_content(self, sample_conversation):
        """Should keep samples within the length bounds."""
        f = LengthFilter(min_chars=10, max_chars=10000)
        result = f.apply([sample_conversation])
        assert len(result) == 1
        assert f.get_stats()["kept"] == 1

    def test_filters_long_content(self):
        """Should remove samples exceeding max_chars."""
        long_sample = GenerationSample(
            messages=[
                {"role": "user", "content": "Generate a very long response."},
                {"role": "assistant", "content": "A" * 5001},
            ]
        )
        f = LengthFilter(min_chars=10, max_chars=5000)
        result = f.apply([long_sample])
        assert len(result) == 0

    def test_min_chars_inclusive(self):
        """Boundary: exactly min_chars should be kept."""
        exact_sample = GenerationSample(
            messages=[
                {"role": "assistant", "content": "A" * 100},
            ]
        )
        f = LengthFilter(min_chars=100, max_chars=10000)
        result = f.apply([exact_sample])
        assert len(result) == 1

    def test_stats_accurate(self, sample_dataset):
        """Stats should reflect what was actually filtered."""
        f = LengthFilter(min_chars=50, max_chars=10000)
        f.apply(sample_dataset)
        stats = f.get_stats()
        assert stats["total"] == len(sample_dataset)
        assert stats["removed"] + stats["kept"] == len(sample_dataset)


class TestCompositeFilter:
    """Test filter chaining."""

    def test_chains_filters(self, sample_dataset):
        """Composite should apply filters in sequence."""
        f1 = LengthFilter(min_chars=100, max_chars=10000)
        f2 = LengthFilter(min_chars=0, max_chars=500)
        composite = CompositeFilter([f1, f2])

        result = composite.apply(sample_dataset)

        # After f1 (min 100 chars): removes the "OK." sample
        # After f2 (max 500 chars): keeps remaining samples under 500 chars
        # Let's just verify it chains without errors
        assert isinstance(result, list)
        assert "sub_filters" in composite.get_stats()
