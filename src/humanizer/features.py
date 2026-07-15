"""
Statistical features for distinguishing human vs AI-generated text.

Human text has distinctive statistical signatures that AI text lacks:
- Higher perplexity variance (burstiness)
- More structural variability in sentence length
- Idiosyncratic transitions and discourse markers
- Higher lexical diversity in certain dimensions
- Natural error distributions (typos, informal grammar)
"""

import re
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextFeatures:
    """Statistical features extracted from text for human/AI classification."""
    
    # Basic counts
    total_chars: int = 0
    total_words: int = 0
    total_sentences: int = 0
    total_paragraphs: int = 0
    
    # Sentence-level features
    sentence_lengths: list[int] = field(default_factory=list)
    sentence_length_mean: float = 0.0
    sentence_length_std: float = 0.0
    sentence_length_cv: float = 0.0  # Coefficient of variation
    sentence_length_entropy: float = 0.0  # Shannon entropy of sentence length distribution
    
    # Word-level features
    avg_word_length: float = 0.0
    word_length_std: float = 0.0
    type_token_ratio: float = 0.0  # Unique words / total words
    hapax_legomena_ratio: float = 0.0  # Words appearing exactly once
    hapax_dislegomena_ratio: float = 0.0  # Words appearing twice
    
    # Perplexity proxies (without actual LLM)
    # These approximate what perplexity-based detectors measure
    punctuation_variance: float = 0.0  # How predictable is punctuation placement
    bigram_repetition_rate: float = 0.0  # How often do bigrams repeat (AI text repeats more)
    trigram_repetition_rate: float = 0.0
    
    # Structural features
    paragraph_length_mean: float = 0.0
    paragraph_length_std: float = 0.0
    sentence_complexity_mean: float = 0.0  # Clauses per sentence
    sentence_complexity_std: float = 0.0
    
    # Stylistic features
    transition_word_ratio: float = 0.0  # "however", "therefore", "moreover", etc.
    first_person_ratio: float = 0.0  # I, me, my, we, our
    second_person_ratio: float = 0.0  # you, your
    contraction_ratio: float = 0.0  # don't, can't, it's, etc.
    informal_word_ratio: float = 0.0  # kinda, gonna, pretty (as adverb), actually
    hedge_ratio: float = 0.0  # maybe, perhaps, seems, appears, somewhat
    
    # AI-typical patterns (higher in AI text)
    ai_template_ratio: float = 0.0  # "it is important to note", "furthermore", "in conclusion"
    balanced_structure_score: float = 0.0  # AI likes balanced "on one hand / on the other hand"
    enumeration_score: float = 0.0  # "first", "second", "third" patterns
    definitive_ratio: float = 0.0  # "clearly", "undoubtedly", "certainly", "obviously"
    
    # Readability
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0


class FeatureExtractor:
    """Extract statistical features from text for human/AI classification."""
    
    # AI-typical transition words (overused by LLMs)
    AI_TRANSITIONS = {
        "furthermore", "moreover", "consequently", "nevertheless", "nonetheless",
        "in conclusion", "to summarize", "it is worth noting", "it is important to note",
        "additionally", "in addition", "as a result", "on the other hand",
        "in contrast", "similarly", "specifically", "particularly",
    }
    
    # Human-typical transitions
    HUMAN_TRANSITIONS = {
        "but", "so", "and", "then", "though", "anyway", "actually",
        "I mean", "you know", "like", "well", "okay", "right",
    }
    
    # Human-typical informal words
    INFORMAL_WORDS = {
        "kinda", "sorta", "gonna", "wanna", "pretty", "actually",
        "basically", "literally", "honestly", "anyway", "whatever",
        "stuff", "thing", "things", "really", "just",
    }
    
    # Contractions
    CONTRACTIONS_PATTERN = re.compile(
        r"\b(?:don't|can't|won't|shouldn't|wouldn't|couldn't|isn't|aren't|wasn't|"
        r"weren't|hasn't|haven't|hadn't|doesn't|didn't|it's|that's|there's|"
        r"what's|who's|he's|she's|I'm|you're|we're|they're|I'll|you'll|"
        r"he'll|she'll|we'll|they'll|I'd|you'd|he'd|she'd|we'd|they'd|"
        r"I've|you've|we've|they've|let's)\b"
    )
    
    # Enumeration patterns (AI loves these)
    ENUMERATION_PATTERN = re.compile(
        r'\b(?:firstly|secondly|thirdly|fourthly|fifthly|'
        r'first[, ]|second[, ]|third[, ]|fourth[, ]|fifth[, ]|'
        r'\d+[.)]\s)', re.IGNORECASE
    )
    
    # Hedge words
    HEDGE_WORDS = {
        "maybe", "perhaps", "seems", "appears", "somewhat", "rather",
        "quite", "fairly", "relatively", "presumably", "possibly",
        "likely", "unlikely",
    }
    
    # AI template phrases
    AI_TEMPLATES = [
        "it is important to note",
        "it is worth mentioning",
        "it should be noted",
        "it is essential to",
        "it is crucial to",
        "plays a crucial role",
        "has been shown to",
        "research has shown",
        "studies have demonstrated",
        "a wide range of",
        "in the context of",
        "in terms of",
        "from this perspective",
        "it can be argued that",
        "there is no doubt that",
    ]
    
    def extract(self, text: str) -> TextFeatures:
        """Extract all statistical features from text."""
        features = TextFeatures()
        
        # Clean and tokenize
        text = text.strip()
        if not text:
            return features
        
        # Paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        features.total_paragraphs = len(paragraphs)
        
        # Sentences
        sentences = self._split_sentences(text)
        features.total_sentences = len(sentences)
        features.sentence_lengths = [len(s.split()) for s in sentences]
        
        # Words
        words = self._tokenize_words(text)
        features.total_words = len(words)
        features.total_chars = len(text)
        
        if not words:
            return features
        
        # Sentence length statistics
        if features.sentence_lengths:
            features.sentence_length_mean = statistics.mean(features.sentence_lengths)
            if len(features.sentence_lengths) > 1:
                features.sentence_length_std = statistics.stdev(features.sentence_lengths)
                features.sentence_length_cv = (
                    features.sentence_length_std / features.sentence_length_mean 
                    if features.sentence_length_mean > 0 else 0
                )
            features.sentence_length_entropy = self._entropy(features.sentence_lengths)
        
        # Word length
        word_lengths = [len(w) for w in words]
        features.avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        features.word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
        
        # Lexical diversity
        word_counts = Counter(w.lower() for w in words)
        total_lower = sum(word_counts.values())
        features.type_token_ratio = len(word_counts) / total_lower if total_lower > 0 else 0
        
        # Hapax legomena (words appearing only once)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        features.hapax_legomena_ratio = hapax_count / total_lower if total_lower > 0 else 0
        
        dis_count = sum(1 for count in word_counts.values() if count == 2)
        features.hapax_dislegomena_ratio = dis_count / total_lower if total_lower > 0 else 0
        
        # Bigram/trigram repetition (AI text repeats n-grams more)
        if len(words) > 2:
            bigrams = [f"{words[i].lower()}_{words[i+1].lower()}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            features.bigram_repetition_rate = (
                1.0 - len(bigram_counts) / len(bigrams) if bigrams else 0
            )
        
        if len(words) > 3:
            trigrams = [f"{words[i].lower()}_{words[i+1].lower()}_{words[i+2].lower()}" 
                       for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            features.trigram_repetition_rate = (
                1.0 - len(trigram_counts) / len(trigrams) if trigrams else 0
            )
        
        # Punctuation variance
        punct_pattern = re.compile(r'[.,!?;:]')
        punct_positions = [i for i, c in enumerate(text) if punct_pattern.match(c)]
        if len(punct_positions) > 1:
            intervals = [punct_positions[i+1] - punct_positions[i] 
                        for i in range(len(punct_positions)-1)]
            features.punctuation_variance = statistics.stdev(intervals) if len(intervals) > 1 else 0
        
        # Paragraph length
        para_lengths = [len(self._tokenize_words(p)) for p in paragraphs]
        if para_lengths:
            features.paragraph_length_mean = statistics.mean(para_lengths)
            features.paragraph_length_std = statistics.stdev(para_lengths) if len(para_lengths) > 1 else 0
        
        # Sentence complexity (clauses per sentence, approximated by comma count)
        clause_counts = [s.count(',') + s.count(';') + 1 for s in sentences]
        features.sentence_complexity_mean = statistics.mean(clause_counts) if clause_counts else 0
        features.sentence_complexity_std = statistics.stdev(clause_counts) if len(clause_counts) > 1 else 0
        
        # Stylistic ratios
        lower_text = text.lower()
        features.transition_word_ratio = self._ratio(lower_text, self.AI_TRANSITIONS, words)
        features.first_person_ratio = self._count_pattern(
            r'\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves)\b', text
        ) / len(words) if words else 0
        features.second_person_ratio = self._count_pattern(
            r'\b(?:you|your|yours|yourself|yourselves)\b', text
        ) / len(words) if words else 0
        features.contraction_ratio = len(self.CONTRACTIONS_PATTERN.findall(text)) / len(words) if words else 0
        features.informal_word_ratio = self._ratio(lower_text, self.INFORMAL_WORDS, words)
        features.hedge_ratio = self._ratio(lower_text, self.HEDGE_WORDS, words)
        features.ai_template_ratio = sum(
            lower_text.count(t) for t in self.AI_TEMPLATES
        ) / len(words) if words else 0
        features.enumeration_score = len(self.ENUMERATION_PATTERN.findall(text)) / len(sentences) if sentences else 0
        features.definitive_ratio = self._count_pattern(
            r'\b(?:clearly|undoubtedly|certainly|obviously|definitely|absolutely|indeed)\b', 
            text
        ) / len(words) if words else 0
        
        # Balanced structure score (AI loves "on one hand... on the other hand")
        balanced_patterns = [
            r'on\s+(the\s+)?one\s+hand.*on\s+(the\s+)?other\s+hand',
            r'not\s+only.*but\s+also',
            r'either.*or',
            r'both.*and',
        ]
        features.balanced_structure_score = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in balanced_patterns
        ) / len(sentences) if sentences else 0
        
        # Readability
        features.flesch_reading_ease = self._flesch_reading_ease(text, sentences, words)
        features.flesch_kincaid_grade = self._flesch_kincaid_grade(text, sentences, words)
        
        return features
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r'\b[a-zA-Z]+\b', text)
    
    def _entropy(self, values: list[int]) -> float:
        """Calculate Shannon entropy of a value distribution."""
        if not values:
            return 0.0
        counter = Counter(values)
        total = len(values)
        return -sum((c/total) * math.log2(c/total) for c in counter.values())
    
    def _ratio(self, text: str, word_set: set[str], words: list[str]) -> float:
        """Calculate ratio of words from a set appearing in text."""
        if not words:
            return 0.0
        count = sum(1 for w in words if w.lower() in word_set)
        return count / len(words)
    
    def _count_pattern(self, pattern: str, text: str) -> int:
        """Count regex pattern matches."""
        return len(re.findall(pattern, text))
    
    def _flesch_reading_ease(self, text: str, sentences: list[str], words: list[str]) -> float:
        """Calculate Flesch Reading Ease score."""
        if not sentences or not words:
            return 0.0
        total_syllables = self._estimate_syllables(text)
        return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))
    
    def _flesch_kincaid_grade(self, text: str, sentences: list[str], words: list[str]) -> float:
        """Calculate Flesch-Kincaid Grade Level."""
        if not sentences or not words:
            return 0.0
        total_syllables = self._estimate_syllables(text)
        return 0.39 * (len(words) / len(sentences)) + 11.8 * (total_syllables / len(words)) - 15.59
    
    def _estimate_syllables(self, text: str) -> int:
        """Rough syllable count estimation."""
        words = self._tokenize_words(text)
        count = 0
        for word in words:
            word_lower = word.lower()
            if len(word_lower) <= 3:
                count += 1
            else:
                # Count vowel groups
                vowels = "aeiouy"
                prev_is_vowel = False
                for char in word_lower:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        count += 1
                    prev_is_vowel = is_vowel
                # Adjust for silent 'e'
                if word_lower.endswith('e') and not word_lower.endswith('le'):
                    count -= 1
                if count == 0:
                    count = 1
        return count


def feature_vector(features: TextFeatures) -> dict[str, float]:
    """Convert TextFeatures to a numeric feature vector for classification."""
    return {
        "sentence_length_mean": float(features.sentence_length_mean),
        "sentence_length_std": float(features.sentence_length_std),
        "sentence_length_cv": float(features.sentence_length_cv),
        "sentence_length_entropy": float(features.sentence_length_entropy),
        "avg_word_length": float(features.avg_word_length),
        "word_length_std": float(features.word_length_std),
        "type_token_ratio": float(features.type_token_ratio),
        "hapax_legomena_ratio": float(features.hapax_legomena_ratio),
        "hapax_dislegomena_ratio": float(features.hapax_dislegomena_ratio),
        "punctuation_variance": float(features.punctuation_variance),
        "bigram_repetition_rate": float(features.bigram_repetition_rate),
        "trigram_repetition_rate": float(features.trigram_repetition_rate),
        "paragraph_length_mean": float(features.paragraph_length_mean),
        "paragraph_length_std": float(features.paragraph_length_std),
        "sentence_complexity_mean": float(features.sentence_complexity_mean),
        "sentence_complexity_std": float(features.sentence_complexity_std),
        "transition_word_ratio": float(features.transition_word_ratio),
        "first_person_ratio": float(features.first_person_ratio),
        "contraction_ratio": float(features.contraction_ratio),
        "informal_word_ratio": float(features.informal_word_ratio),
        "hedge_ratio": float(features.hedge_ratio),
        "ai_template_ratio": float(features.ai_template_ratio),
        "balanced_structure_score": float(features.balanced_structure_score),
        "enumeration_score": float(features.enumeration_score),
        "definitive_ratio": float(features.definitive_ratio),
        "flesch_reading_ease": float(features.flesch_reading_ease),
        "flesch_kincaid_grade": float(features.flesch_kincaid_grade),
    }