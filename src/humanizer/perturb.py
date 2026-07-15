"""
Targeted text perturbation engine.

Instead of blindly paraphrasing (which introduces its own detectable patterns),
this applies surgical modifications to the specific features that trigger AI detection,
while preserving meaning and readability.
"""

import random
import re
from typing import Optional

from .features import FeatureExtractor, TextFeatures
from .benchmark import HeuristicDetector


class PerturbationStrategy:
    """Individual perturbation strategy with a specific goal."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def apply(self, text: str) -> str:
        raise NotImplementedError


class SentenceLengthVariation(PerturbationStrategy):
    """Vary sentence lengths to increase burstiness.
    
    AI text tends toward uniform sentence lengths. This splits long
    sentences and merges short ones to create natural variation.
    """
    
    def __init__(self):
        super().__init__(
            "sentence_length_variation",
            "Vary sentence lengths to create natural rhythm"
        )
    
    def apply(self, text: str) -> str:
        extractor = FeatureExtractor()
        sentences = extractor._split_sentences(text)
        
        if len(sentences) < 3:
            return text
        
        result = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            words = extractor._tokenize_words(sentence)
            
            # Split very long sentences (30+ words)
            if len(words) > 30 and random.random() < 0.4:
                mid = len(words) // 2
                # Find a good split point near middle (after comma, semicolon, or conjunction)
                split_words = sentence.split()
                split_pos = mid
                for j in range(mid - 5, mid + 5):
                    if j < len(split_words) and split_words[j].rstrip(',;') in ('and', 'but', 'or', 'so', 'while', 'because', 'although', 'however'):
                        split_pos = j + 1
                        break
                
                part1 = ' '.join(split_words[:split_pos]).rstrip(',;') + '.'
                part2 = ' '.join(split_words[split_pos:])
                if part2 and part2[0].islower():
                    part2 = part2[0].upper() + part2[1:]
                result.append(part1)
                if part2.strip():
                    result.append(part2)
            
            # Merge very short consecutive sentences (under 8 words)
            elif len(words) < 8 and i + 1 < len(sentences):
                next_words = extractor._tokenize_words(sentences[i + 1])
                if len(next_words) < 15 and random.random() < 0.3:
                    merged = sentence.rstrip('.!?') + ', ' + sentences[i + 1][0].lower() + sentences[i + 1][1:]
                    result.append(merged)
                    i += 1  # Skip next
                else:
                    result.append(sentence)
            else:
                result.append(sentence)
            i += 1
        
        return ' '.join(result)


class ContractionInsertion(PerturbationStrategy):
    """Insert natural contractions where appropriate."""
    
    CONTRACTIONS = [
        ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
        ("cannot", "can't"), ("will not", "won't"), ("would not", "wouldn't"),
        ("should not", "shouldn't"), ("could not", "couldn't"),
        ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
        ("were not", "weren't"), ("has not", "hasn't"), ("have not", "haven't"),
        ("had not", "hadn't"), ("it is", "it's"), ("that is", "that's"),
        ("there is", "there's"), ("I am", "I'm"), ("you are", "you're"),
        ("we are", "we're"), ("they are", "they're"),
        ("I will", "I'll"), ("you will", "you'll"), ("we will", "we'll"),
        ("they will", "they'll"), ("I would", "I'd"), ("you would", "you'd"),
        ("I have", "I've"), ("you have", "you've"), ("we have", "we've"),
    ]
    
    def __init__(self):
        super().__init__(
            "contraction_insertion",
            "Insert natural contractions (don't, can't, it's)"
        )
    
    def apply(self, text: str) -> str:
        result = text
        for full, contracted in self.CONTRACTIONS:
            # Only replace ~30% of instances to keep it natural
            if full in result and random.random() < 0.3:
                result = result.replace(full, contracted, 1)
        return result


class TransitionReduction(PerturbationStrategy):
    """Reduce AI-typical transition words."""
    
    AI_TRANSITIONS = [
        "furthermore", "moreover", "nevertheless", "nonetheless",
        "consequently", "additionally", "in addition",
        "in conclusion", "to summarize", "it is worth noting",
        "it is important to note", "specifically", "particularly",
    ]
    
    REPLACEMENTS = {
        "furthermore": "also",
        "moreover": "also",
        "nevertheless": "but",
        "nonetheless": "still",
        "consequently": "so",
        "additionally": "also",
        "in addition": "plus",
    }
    
    def __init__(self):
        super().__init__(
            "transition_reduction",
            "Reduce AI-typical transition words"
        )
    
    def apply(self, text: str) -> str:
        result = text
        lower = result.lower()
        for transition in self.AI_TRANSITIONS:
            if transition in lower and random.random() < 0.5:
                replacement = self.REPLACEMENTS.get(transition, "")
                if replacement:
                    # Case-preserving replacement
                    pattern = re.compile(re.escape(transition), re.IGNORECASE)
                    result = pattern.sub(replacement, result, count=1)
                else:
                    # Remove entirely
                    pattern = re.compile(r'\s*' + re.escape(transition) + r',?\s*', re.IGNORECASE)
                    result = pattern.sub(' ', result)
        return result


class InformalInjection(PerturbationStrategy):
    """Inject natural informal language patterns."""
    
    INFORMAL_INJECTIONS = [
        ("actually", 0.1), ("pretty", 0.1), ("really", 0.1),
        ("just", 0.1), ("kind of", 0.05), ("sort of", 0.05),
    ]
    
    def __init__(self):
        super().__init__(
            "informal_injection",
            "Add natural informal language (actually, pretty, just)"
        )
    
    def apply(self, text: str) -> str:
        sentences = FeatureExtractor()._split_sentences(text)
        result = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 4:
                result.append(sentence)
                continue
            
            modified = sentence
            for word, prob in self.INFORMAL_INJECTIONS:
                if random.random() < prob and word not in modified.lower():
                    # Insert near the beginning of the sentence
                    insert_pos = random.randint(2, min(5, len(words) - 1))
                    words = modified.split()
                    words.insert(insert_pos, word)
                    modified = ' '.join(words)
                    break
            
            result.append(modified)
        
        return ' '.join(result)


class SentenceStartVariation(PerturbationStrategy):
    """Vary sentence starts to avoid AI's repetitive patterns.
    
    AI text often starts sentences with "The", "This", "It", or "These".
    Human text has more varied sentence openings.
    """
    
    def __init__(self):
        super().__init__(
            "sentence_start_variation",
            "Vary sentence openings to avoid AI patterns"
        )
    
    def apply(self, text: str) -> str:
        sentences = FeatureExtractor()._split_sentences(text)
        result = []
        
        for sentence in sentences:
            if not sentence:
                result.append(sentence)
                continue
            
            # Occasionally restructure "The X is Y" → "Y is what X does"
            # or "This suggests that" → "What this suggests is that"
            if random.random() < 0.15:
                words = sentence.split()
                if len(words) > 6:
                    if words[0].lower() == 'the' and ' is ' in sentence.lower():
                        # Simple inversion
                        pass  # Keep as-is for now; inversions risk grammar errors
                    elif words[0].lower() == 'this' and len(words) > 3:
                        words[0] = random.choice(["What this", "That", "It"])
                        sentence = ' '.join(words)
                    elif words[0].lower() == 'it' and len(words) > 4:
                        sentence = sentence[3:].strip()  # Drop "It "
            
            result.append(sentence)
        
        return ' '.join(result)


class ParagraphBreakup(PerturbationStrategy):
    """Break up uniform paragraph structures."""
    
    def __init__(self):
        super().__init__(
            "paragraph_breakup",
            "Break uniform paragraph structures"
        )
    
    def apply(self, text: str) -> str:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if len(paragraphs) <= 1:
            return text
        
        result = []
        for i, para in enumerate(paragraphs):
            sentences = FeatureExtractor()._split_sentences(para)
            
            # If paragraph is long and even-numbered, split it
            if len(sentences) > 4 and i % 2 == 0 and random.random() < 0.3:
                mid = len(sentences) // 2
                result.append(' '.join(sentences[:mid]))
                result.append('')
                result.append(' '.join(sentences[mid:]))
            else:
                result.append(para)
            
            if i < len(paragraphs) - 1:
                result.append('')
        
        return '\n\n'.join(r for r in result if r)


class Humanizer:
    """Applies perturbation strategies to reduce AI detection scores."""
    
    def __init__(self, strategies: Optional[list[PerturbationStrategy]] = None):
        self.strategies = strategies or self._default_strategies()
        self.detector = HeuristicDetector()
        self.extractor = FeatureExtractor()
    
    def _default_strategies(self) -> list[PerturbationStrategy]:
        return [
            SentenceLengthVariation(),
            ContractionInsertion(),
            TransitionReduction(),
            InformalInjection(),
            SentenceStartVariation(),
            ParagraphBreakup(),
        ]
    
    def humanize(self, text: str, max_iterations: int = 3) -> str:
        """Apply perturbation strategies to reduce AI detection.
        
        Iteratively applies strategies and checks if detection score improves.
        Stops when AI probability drops below threshold or max iterations reached.
        """
        current = text
        best_text = text
        ai_prob, _ = self.detector.detect(text)
        best_score = ai_prob
        
        for iteration in range(max_iterations):
            improved = False
            
            for strategy in self.strategies:
                candidate = strategy.apply(current)
                if candidate == current:
                    continue
                
                new_ai_prob, _ = self.detector.detect(candidate)
                
                if new_ai_prob < best_score:
                    best_score = new_ai_prob
                    best_text = candidate
                    improved = True
            
            if not improved or best_score < 0.3:
                break
            
            current = best_text
        
        return best_text
    
    def humanize_with_report(self, text: str) -> dict:
        """Humanize text and return before/after comparison."""
        before_ai_prob, before_details = self.detector.detect(text)
        humanized = self.humanize(text)
        after_ai_prob, after_details = self.detector.detect(humanized)
        
        return {
            "original_text": text,
            "humanized_text": humanized,
            "before_ai_probability": before_ai_prob,
            "after_ai_probability": after_ai_prob,
            "improvement": before_ai_prob - after_ai_prob,
            "before_details": before_details,
            "after_details": after_details,
            "strategies_applied": [s.name for s in self.strategies],
            "characters_changed": sum(1 for a, b in zip(text, humanized) if a != b) + abs(len(text) - len(humanized)),
        }