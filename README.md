# Humanizer

**AI text humanization with statistical proof — the one that actually works**

Most "AI humanizers" are wrappers around another LLM with a "make this sound human" prompt — they fail against any serious detector. Humanizer takes a different approach: statistical modeling of human vs. AI writing patterns, benchmarked rigorously against leading detectors.

## The Problem

AI-generated text has detectable statistical signatures — predictable token distributions, low perplexity variance, structural templates. Simple paraphrasing doesn't fix this because it introduces its own detectable patterns. Real humanization requires understanding *why* detectors flag content and surgically modifying those signals.

## Approach (planned)

1. **Train a detector** — understand the enemy. Build a classifier trained on human vs AI text across domains
2. **Identify features** — what separates human from AI: perplexity bursts, structural variability, idiosyncratic transitions, error distributions
3. **Targeted perturbation** — not blind paraphrasing. Modify only the features that trigger detection, preserving meaning
4. **Statistical validation** — benchmark against GPTZero, Originality.ai, Turnitin, ZeroGPT, and others with published results
5. **Iterate adversarially** — each new detector release becomes training data

## Planned Features

- **Multi-detector benchmark suite** — automated testing against 10+ detectors with published scores
- **Feature-level control** — adjust specific dimensions: perplexity variance, sentence length entropy, transition diversity
- **Domain-specific models** — academic, business, creative, technical each have different human signatures
- **Statistical proof report** — every output comes with detection probability across all major detectors
- **API and CLI** — integrate into any workflow

## Success Criteria

- **>95% human classification** on GPTZero, Originality.ai, Turnitin simultaneously
- **Meaning preservation** — BERTScore > 0.95 vs original
- **Published methodology** — no black boxes, fully reproducible benchmarks

## Tech Stack (planned)

- Python (core pipeline, statistical modeling)
- Custom perplexity analysis
- Fine-tuned perturbation models
- Automated benchmark harness against commercial detectors

## Status

🚧 Research phase — corpus collection, detector analysis, and feature engineering.

## This Is Not

- ❌ A wrapper around GPT-4 with "make it sound human"
- ❌ A synonym-replacement spinner
- ❌ A black box with unverifiable claims