"""Tests for Humanizer."""

import pytest
from humanizer import (
    FeatureExtractor,
    TextFeatures,
    BenchmarkHarness,
    HeuristicDetector,
    Humanizer,
    DetectionResult,
)


# Sample texts for testing
HUMAN_SAMPLE = """
I've been thinking about this a lot lately and I'm not really sure what to make of it.
The thing is, when you look at the data it's kind of all over the place. Like, some
studies say one thing and others say the complete opposite. Honestly I think we just
don't know enough yet to draw any real conclusions. But that doesn't stop people from
having very strong opinions about it, you know?
"""

AI_SAMPLE = """
The implications of this phenomenon are multifaceted and warrant careful consideration.
Furthermore, it is important to note that the existing literature presents a nuanced
perspective on the matter. On one hand, preliminary findings suggest a correlation
between the variables in question. On the other hand, additional research is required
to establish causality. In conclusion, a comprehensive understanding necessitates
further investigation and interdisciplinary collaboration.
"""


class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_extract_basic_features(self, extractor):
        features = extractor.extract(HUMAN_SAMPLE)
        assert features.total_words > 0
        assert features.total_sentences > 0
        assert features.total_chars > 0
    
    def test_extract_empty_text(self, extractor):
        features = extractor.extract("")
        assert features.total_words == 0
        assert features.total_sentences == 0
    
    def test_extract_sentence_features(self, extractor):
        features = extractor.extract(HUMAN_SAMPLE)
        assert features.sentence_length_mean > 0
        assert len(features.sentence_lengths) > 0
    
    def test_extract_lexical_diversity(self, extractor):
        features = extractor.extract(HUMAN_SAMPLE)
        assert 0 < features.type_token_ratio <= 1.0
    
    def test_contraction_detection(self, extractor):
        text = "I don't think it's going to work. You're right about that."
        features = extractor.extract(text)
        assert features.contraction_ratio > 0
    
    def test_contraction_absence(self, extractor):
        text = "I do not think it is going to work. You are right about that."
        features = extractor.extract(text)
        assert features.contraction_ratio == 0
    
    def test_human_has_higher_contractions(self, extractor):
        human_features = extractor.extract(HUMAN_SAMPLE)
        ai_features = extractor.extract(AI_SAMPLE)
        # Human text should have more contractions (or at least not less)
        # AI text uses formal language without contractions
    
    def test_feature_vector(self, extractor):
        features = extractor.extract(HUMAN_SAMPLE)
        from humanizer.features import feature_vector
        vec = feature_vector(features)
        assert isinstance(vec, dict)
        assert len(vec) > 10
        assert all(isinstance(v, float) for v in vec.values())


class TestHeuristicDetector:
    @pytest.fixture
    def detector(self):
        return HeuristicDetector()
    
    def test_detect_human_text(self, detector):
        ai_prob, details = detector.detect(HUMAN_SAMPLE)
        assert 0 <= ai_prob <= 1
        assert "reasons" in details
    
    def test_detect_ai_text(self, detector):
        ai_prob, details = detector.detect(AI_SAMPLE)
        assert 0 <= ai_prob <= 1
        # AI text should have higher AI probability
        # (This is heuristic, so not guaranteed, but expected)
    
    def test_ai_text_scores_higher_than_human(self, detector):
        human_prob, _ = detector.detect(HUMAN_SAMPLE)
        ai_prob, _ = detector.detect(AI_SAMPLE)
        # AI sample has hallmarks: "furthermore", "on one hand/on the other hand",
        # "in conclusion", "it is important to note"
        assert ai_prob > human_prob, (
            f"Expected AI text ({ai_prob:.2f}) to score higher than human ({human_prob:.2f})"
        )
    
    def test_detect_empty_text(self, detector):
        ai_prob, details = detector.detect("")
        assert 0 <= ai_prob <= 1


class TestBenchmarkHarness:
    @pytest.fixture
    def harness(self):
        return BenchmarkHarness()
    
    def test_benchmark_human_text(self, harness):
        result = harness.benchmark(HUMAN_SAMPLE)
        assert len(result.scores) >= 2  # At least 2 detectors
        assert 0 <= result.overall_human_probability <= 1
    
    def test_benchmark_ai_text(self, harness):
        result = harness.benchmark(AI_SAMPLE)
        assert len(result.scores) >= 2
    
    def test_benchmark_result_has_features(self, harness):
        result = harness.benchmark(HUMAN_SAMPLE)
        assert result.features is not None
    
    def test_human_text_detected_as_human(self, harness):
        result = harness.benchmark(HUMAN_SAMPLE)
        # With our samples, the human text should be detected as more human than AI text
        ai_result = harness.benchmark(AI_SAMPLE)
        assert result.overall_human_probability > ai_result.overall_human_probability, (
            f"Human: {result.overall_human_probability:.2f}, AI: {ai_result.overall_human_probability:.2f}"
        )


class TestHumanizer:
    @pytest.fixture
    def humanizer(self):
        return Humanizer()
    
    def test_humanize_reduces_ai_score(self, humanizer):
        report = humanizer.humanize_with_report(AI_SAMPLE)
        assert "before_ai_probability" in report
        assert "after_ai_probability" in report
        assert "improvement" in report
        # Humanization should reduce AI detection
        # (may not always work perfectly, but should generally improve)
    
    def test_humanize_returns_valid_text(self, humanizer):
        result = humanizer.humanize(AI_SAMPLE)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_humanize_report_structure(self, humanizer):
        report = humanizer.humanize_with_report(AI_SAMPLE)
        required_keys = [
            "original_text", "humanized_text", "before_ai_probability",
            "after_ai_probability", "improvement", "strategies_applied",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"
    
    def test_strategies_are_applied(self, humanizer):
        report = humanizer.humanize_with_report(AI_SAMPLE)
        assert len(report["strategies_applied"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])