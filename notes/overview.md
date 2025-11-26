# Nested Learning notes

## What is Nested Learning?
Nested Learning (NL) treats a model as a stack of interconnected optimization problems that run at different time scales. Each component—optimizer states, attention memories, or auxiliary modules—has its own **context flow** (the information it tries to compress) and its own update frequency. Layers that update faster are nested inside slower ones, yielding a hierarchy reminiscent of multi-frequency brain waves highlighted in the Google Research blog post.

Key ideas from the paper and blog:
- Architecture and optimizer are two views of the same thing: each level is an associative memory that learns from its context flow and hands compressed signals to slower levels.
- Standard components can be reinterpreted: momentum is a value-less associative memory, preconditioning adds key–value capacity, and attention blocks are non-parametric associative memories with their own fast context streams.
- Designing models means choosing how many nested levels to expose and how frequently each level updates, enabling multi-time-scale continual learning without catastrophic forgetting.

## How does the HOPE architecture apply NL?
The paper introduces **HOPE**, a self-referential learning module built by wiring multiple NL levels together (e.g., deep momentum, associative preconditioning, and attention viewed as fast memories). HOPE reuses a uniform block structure but allows every block to update at its own frequency, giving the model a continuum memory that can adapt quickly while keeping slower weights stable.

## Reported empirical gains
NL is evaluated through HOPE on language modeling and commonsense reasoning benchmarks. Highlights from Table 1 of the paper include:

| Model scale | Training tokens | Best baseline (Avg) | HOPE Avg | Notes |
| --- | --- | --- | --- | --- |
| 1.3B params | 100B | Titans (LMM) – 51.56 | **52.26** | HOPE edges out both Titans and Samba hybrids on the mixed LM/QA suite. |
| Larger scale block | (unspecified in text snippet) | Titans (LMM) – 56.82 | **57.23** | HOPE achieves the top average score across the reported tasks. |

Even at smaller scales (760M params, 30B tokens), HOPE achieves competitive averages alongside Transformer++, RetNet, DeltaNet, and Samba, while using the same uniform building block across levels.

## Mental model for implementation
1. **Define levels and frequencies.** Decide which states update every token (fast memory), every step (optimizer states), or every few steps (slow weights).
2. **Expose context flows.** Each level receives the context it will compress—gradients for optimizer states, token streams for attention, or hidden activations for deep momentum memories.
3. **Share block definitions.** Use a reusable block that can act as a fast associative memory or a slower parameter buffer depending on its update rate.
4. **Train jointly.** Optimize all levels simultaneously; faster levels adapt within sequences, while slower levels accumulate long-term knowledge.

See `examples/nested_learning.py` for a minimal PyTorch-style sketch of this pattern.
