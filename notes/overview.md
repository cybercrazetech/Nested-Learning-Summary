# Nested Learning notes

## What is Nested Learning?
Nested Learning (NL) treats a model as a stack of interconnected optimization problems that run at different time scales. Each component—optimizer states, attention memories, or auxiliary modules—has its own **context flow** (the information it tries to compress) and its own update frequency. Layers that update faster are nested inside slower ones, yielding a hierarchy reminiscent of multi-frequency brain waves highlighted in the Google Research blog post.

Key ideas from the paper and blog:
- Architecture and optimizer are two views of the same thing: each level is an associative memory that learns from its context flow and hands compressed signals to slower levels.
- Standard components can be reinterpreted: momentum is a value-less associative memory, preconditioning adds key–value capacity, and attention blocks are non-parametric associative memories with their own fast context streams.
- Designing models means choosing how many nested levels to expose and how frequently each level updates, enabling multi-time-scale continual learning without catastrophic forgetting.

## Mental picture: how the levels fit together
A simplified view of the continuum memory used in the paper’s HOPE module:

```
[token stream] -> fast attention memory (updates every token)
             \-> deep momentum / optimizer state (updates every step)
                 \-> slow parameters (update every few steps)
```

- **Fast attention memory** handles token-level adaptation (keys/values refreshed per token), akin to a scratchpad.
- **Deep momentum / associative preconditioner** compresses gradients every optimization step, acting as a mid-speed memory.
- **Slow parameters** integrate what slower context remains after fast memories have adapted, anchoring long-term knowledge.

## How HOPE is structured
HOPE wires these levels using the same building block repeated at different frequencies. Figure 3 of the paper contrasts HOPE with a Transformer: HOPE removes per-layer normalization and instead stitches together (1) an MLP-style channel mixer, (2) an attention-style associative memory, and (3) a deep optimizer-style update path. This yields a self-referential module whose fast sub-blocks can alter its slower weights during the forward pass.

A compact structural outline:
```
HOPE block:
  x -> ChannelMixer -> FastMemory(attention-like, token update)
     -> OptimizerMemory(momentum/preconditioner, step update)
     -> SlowWeights (multi-step update)
```
By contrast, a standard Transformer block keeps a single forward path (attention + MLP) and updates weights only via the outer optimizer.

## What Titans is and how it differs
Titans (sometimes shown as Titans LMM in the paper) is a self-modifying sequence model that learns its own update algorithm. It already contains a fast algorithmic path, but NL/HOPE extends this idea into a multi-level continuum: HOPE nests more associative memories (deep momentum, associative preconditioning, attention) and lets them update at distinct rates. Compared with Transformers or Titans, HOPE’s nested levels provide:
- **More granular time scales** (token, step, multi-step) rather than a single optimizer step.
- **Reusable block design**: the same associative-memory block can serve as attention or optimizer memory depending on its update rate.
- **Continual-learning resilience** by refreshing fast memories without overwriting slow weights.

## Reported empirical gains
Table 1 of the paper reports language modeling perplexities and commonsense QA accuracy averages. Highlights:

- **760M params / 30B tokens**: HOPE average score 46.90 vs. Transformer++ (48.69), RetNet (48.46), DeltaNet (48.97), TTT (47.32), Samba* (51.08).【5bae28†L21-L32】【5bae28†L34-L42】
- **1.3B params / 100B tokens**: HOPE average 52.26 vs. Titans (LMM) 51.56 and Samba* 51.08; HOPE edges out the Titans baseline across the mixed LM/QA suite.【5bae28†L13-L20】【5bae28†L34-L42】

(“Samba*” denotes a hybrid model in the table.)

## Implementation walk-through
1. **Define levels and frequencies.** Decide which states update every token (fast memory), every step (optimizer states), or every few steps (slow weights).
2. **Expose context flows.** Each level receives the context it will compress—gradients for optimizer states, token streams for attention, or hidden activations for deep momentum memories.
3. **Share block definitions.** Use a reusable block that can act as a fast associative memory or a slower parameter buffer depending on its update rate.
4. **Train jointly.** Optimize all levels simultaneously; faster levels adapt within sequences, while slower levels accumulate long-term knowledge.

See `examples/nested_learning.py` for a minimal PyTorch-style sketch of this pattern and how it differs from a vanilla Transformer training loop.
