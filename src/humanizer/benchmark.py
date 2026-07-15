"""
Benchmark harness for AI text detectors.

Tests text against multiple detection APIs and reports results.
Currently supports heuristic detection; API integrations planned for:
- GPTZero
- Originality.ai
- Turnitin
- ZeroGPT
- Copyleaks
- Writer.com AI Detector
- Sapling AI Detector
- GLTR (Giant Language model Test Room)
"""

import json
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from .features import FeatureExtractor, TextFeatures, feature_vector


class DetectionResult(Enum):
    HUMAN = "human"
    AI = "ai"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


@dataclass
class DetectorScore:
    """Score from a single detector."""
    detector_name: str
    result: DetectionResult
    confidence: float  # 0.0 - 1.0
    human_probability: float  # 0.0 - 1.0 probability it's human
    ai_probability: float  # 0.0 - 1.0 probability it's AI
    details: dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a text sample."""
    text: str
    text_length: int
    scores: list[DetectorScore]
    overall_human_probability: float = 0.0
    majority_vote: DetectionResult = DetectionResult.UNCERTAIN
    detectors_agree: bool = False
    features: Optional[TextFeatures] = None


class HeuristicDetector:
    """Heuristic AI text detector based on statistical features.
    
    Uses a rules-based approach combining multiple statistical signals
    to estimate the probability that text is AI-generated vs human-written.
    """
    
    def __init__(self):
        self.extractor = FeatureExtractor()
    
    def detect(self, text: str) -> tuple[float, dict]:
        """Return (ai_probability, details).
        
        Returns probability 0-1 that text is AI-generated.
        0 = likely human, 1 = likely AI.
        """
        features = self.extractor.extract(text)
        vec = feature_vector(features)
        
        # Scoring based on known AI vs human patterns
        score = 0.0
        reasons = []
        
        # AI text tends to have more uniform sentence lengths
        if vec["sentence_length_cv"] < 0.4:
            score += 0.15
            reasons.append(f"Low sentence length variation (CV={vec['sentence_length_cv']:.2f})")
        elif vec["sentence_length_cv"] > 0.8:
            score -= 0.10
            reasons.append(f"High sentence length variation (CV={vec['sentence_length_cv']:.2f})")
        
        # AI text has higher bigram repetition
        if vec["bigram_repetition_rate"] > 0.15:
            score += 0.12
            reasons.append(f"High bigram repetition ({vec['bigram_repetition_rate']:.2%})")
        
        # AI text uses more transition words
        if vec["transition_word_ratio"] > 0.03:
            score += 0.12
            reasons.append(f"High transition word usage ({vec['transition_word_ratio']:.2%})")
        elif vec["transition_word_ratio"] < 0.01:
            score -= 0.08
            reasons.append(f"Low transition word usage ({vec['transition_word_ratio']:.2%})")
        
        # AI text uses more definitive language
        if vec["definitive_ratio"] > 0.01:
            score += 0.10
            reasons.append(f"High definitive language ({vec['definitive_ratio']:.2%})")
        
        # AI text has more enumerated structures
        if vec["enumeration_score"] > 0.1:
            score += 0.10
            reasons.append(f"Enumerated structure detected ({vec['enumeration_score']:.2f})")
        
        # AI text tends to have lower type-token ratio (less diverse vocabulary)
        if vec["type_token_ratio"] < 0.4:
            score += 0.08
            reasons.append(f"Low lexical diversity (TTR={vec['type_token_ratio']:.2f})")
        elif vec["type_token_ratio"] > 0.7:
            score -= 0.08
            reasons.append(f"High lexical diversity (TTR={vec['type_token_ratio']:.2f})")
        
        # Human text has more contractions
        if vec["contraction_ratio"] > 0.02:
            score -= 0.10
            reasons.append(f"Natural contraction usage ({vec['contraction_ratio']:.2%})")
        
        # Human text uses more informal words
        if vec["informal_word_ratio"] > 0.02:
            score -= 0.10
            reasons.append(f"Informal language detected ({vec['informal_word_ratio']:.2%})")
        
        # AI text uses more hedging
        if vec["hedge_ratio"] > 0.02:
            score += 0.08
            reasons.append(f"High hedging language ({vec['hedge_ratio']:.2%})")
        
        # AI template phrases
        if vec["ai_template_ratio"] > 0:
            score += 0.15
            reasons.append(f"AI template phrases detected ({vec['ai_template_ratio']:.4f})")
        
        # Balanced structure (AI hallmark)
        if vec["balanced_structure_score"] > 0.05:
            score += 0.10
            reasons.append(f"Balanced structure pattern ({vec['balanced_structure_score']:.2f})")
        
        # Hapax legomena (more in human text)
        if vec["hapax_legomena_ratio"] > 0.5:
            score -= 0.10
            reasons.append(f"High unique word ratio ({vec['hapax_legomena_ratio']:.2%})")
        
        # Clamp to 0-1
        score = max(0.0, min(1.0, score + 0.5))  # Center around 0.5
        
        details = {
            "reasons": reasons,
            "features": {k: round(v, 4) for k, v in vec.items()},
            "raw_score": score,
        }
        
        return score, details


class PerplexityProxy:
    """Proxy for perplexity-based detection without requiring an actual LLM.
    
    Real perplexity-based detectors (like GPTZero) compute perplexity by
    running the text through a language model. This proxy uses n-gram
    statistics and local patterns to approximate the same signal.
    """
    
    def compute_burstiness(self, text: str) -> float:
        """Compute perplexity burstiness - a key GPTZero metric.
        
        Burstiness measures how much sentence perplexity varies.
        Human text has high burstiness (varies a lot). AI text is uniform.
        """
        extractor = FeatureExtractor()
        sentences = extractor._split_sentences(text)
        
        if len(sentences) < 3:
            return 0.0
        
        # Proxy for per-sentence "perplexity": sentence length * word rarity
        words = extractor._tokenize_words(text)
        word_freqs = {}
        for w in words:
            word_freqs[w.lower()] = word_freqs.get(w.lower(), 0) + 1
        
        sentence_scores = []
        for sentence in sentences:
            sent_words = extractor._tokenize_words(sentence)
            if not sent_words:
                continue
            # Higher score = more "perplexing" (lower frequency words)
            rarity = sum(1.0 / word_freqs.get(w.lower(), 1) for w in sent_words) / len(sent_words)
            sentence_scores.append(rarity * len(sent_words))
        
        if len(sentence_scores) < 2:
            return 0.0
        
        mean_score = statistics.mean(sentence_scores)
        if mean_score == 0:
            return 0.0
        
        std_score = statistics.stdev(sentence_scores)
        return std_score / mean_score  # Coefficient of variation = burstiness


class BenchmarkHarness:
    """Runs text through multiple detectors and reports results."""
    
    def __init__(self):
        self.heuristic = HeuristicDetector()
        self.perplexity = PerplexityProxy()
    
    def benchmark(self, text: str) -> BenchmarkResult:
        """Run full benchmark against all available detectors."""
        scores = []
        features = FeatureExtractor().extract(text)
        
        # Heuristic detector
        start = time.time()
        ai_prob, details = self.heuristic.detect(text)
        latency = (time.time() - start) * 1000
        
        human_prob = 1.0 - ai_prob
        
        if ai_prob > 0.65:
            result = DetectionResult.AI
        elif ai_prob < 0.35:
            result = DetectionResult.HUMAN
        elif 0.4 < ai_prob < 0.6:
            result = DetectionResult.MIXED
        else:
            result = DetectionResult.UNCERTAIN
        
        scores.append(DetectorScore(
            detector_name="HeuristicDetector (statistical)",
            result=result,
            confidence=max(ai_prob, human_prob),
            human_probability=human_prob,
            ai_probability=ai_prob,
            details=details,
            latency_ms=latency,
        ))
        
        # Perplexity burstiness check
        burstiness = self.perplexity.compute_burstiness(text)
        if burstiness > 0.6:
            burst_result = DetectionResult.HUMAN
            burst_human_prob = min(0.9, 0.5 + burstiness * 0.5)
        elif burstiness < 0.3:
            burst_result = DetectionResult.AI
            burst_human_prob = max(0.1, 0.5 - burstiness * 0.5)
        else:
            burst_result = DetectionResult.MIXED
            burst_human_prob = 0.5
        
        scores.append(DetectorScore(
            detector_name="BurstinessCheck (perplexity proxy)",
            result=burst_result,
            confidence=max(burst_human_prob, 1 - burst_human_prob),
            human_probability=burst_human_prob,
            ai_probability=1 - burst_human_prob,
            details={"burstiness": burstiness},
            latency_ms=0.5,
        ))
        
        # Aggregate
        human_probs = [s.human_probability for s in scores]
        overall_human = statistics.mean(human_probs) if human_probs else 0.5
        
        votes = [s.result for s in scores]
        if all(v == DetectionResult.HUMAN for v in votes):
            majority = DetectionResult.HUMAN
            agree = True
        elif all(v == DetectionResult.AI for v in votes):
            majority = DetectionResult.AI
            agree = True
        else:
            human_votes = sum(1 for v in votes if v == DetectionResult.HUMAN)
            ai_votes = sum(1 for v in votes if v == DetectionResult.AI)
            if human_votes > ai_votes:
                majority = DetectionResult.HUMAN
            elif ai_votes > human_votes:
                majority = DetectionResult.AI
            else:
                majority = DetectionResult.MIXED
            agree = False
        
        return BenchmarkResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            text_length=len(text),
            scores=scores,
            overall_human_probability=overall_human,
            majority_vote=majority,
            detectors_agree=agree,
            features=features,
        )
    
    def benchmark_batch(self, texts: list[tuple[str, str]]) -> list[BenchmarkResult]:
        """Benchmark multiple texts with labels.
        
        Args:
            texts: List of (text, label) tuples where label is 'human' or 'ai'
        
        Returns:
            List of BenchmarkResult objects
        """
        return [self.benchmark(text) for text, _ in texts]
    
    def accuracy_report(self, results: list[BenchmarkResult], labels: list[str]) -> dict:
        """Generate accuracy report from benchmark results with ground truth labels.
        
        Returns dict with precision, recall, F1, and per-detector stats.
        """
        report = {
            "total_samples": len(results),
            "human_samples": sum(1 for l in labels if l == "human"),
            "ai_samples": sum(1 for l in labels if l == "ai"),
            "detectors": {},
        }
        
        for result in results:
            for score in result.scores:
                if score.detector_name not in report["detectors"]:
                    report["detectors"][score.detector_name] = {
                        "correct": 0, "total": 0, "human_correct": 0,
                        "human_total": 0, "ai_correct": 0, "ai_total": 0,
                    }
        
        for result, label in zip(results, labels):
            is_human = label == "human"
            for score in result.scores:
                det = report["detectors"][score.detector_name]
                det["total"] += 1
                
                predicted_human = score.result == DetectionResult.HUMAN
                
                if is_human:
                    det["human_total"] += 1
                    if predicted_human:
                        det["correct"] += 1
                        det["human_correct"] += 1
                else:
                    det["ai_total"] += 1
                    if not predicted_human:
                        det["correct"] += 1
                        det["ai_correct"] += 1
        
        # Calculate metrics
        for name, stats in report["detectors"].items():
            total = stats["total"]
            stats["accuracy"] = stats["correct"] / total if total > 0 else 0
            stats["human_recall"] = stats["human_correct"] / stats["human_total"] if stats["human_total"] > 0 else 0
            stats["ai_recall"] = stats["ai_correct"] / stats["ai_total"] if stats["ai_total"] > 0 else 0
        
        return report