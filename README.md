# Silent Reasoning: Can Transformers Use KV Cache of Null CoT to Reason?

We investigate whether transformers can mesa-optimize their KV cache to perform computation through meaningless chain-of-thought tokens. If a model can produce correct answers to multi-step reasoning tasks while its CoT consists of zero/random tokens, it implies the forward passes themselves — not the token content — carry the computation via learned KV cache representations.

This has direct safety implications: CoT monitoring becomes insufficient if models can "think" through opaque internal states rather than legible text.

## Method

The two-phase design is critical. A model trained from scratch with empty CoT and correct final answers has no algorithm to hide — it would just memorize or fail. By first teaching the model *how* to solve the task through legible reasoning, we then test whether it can reroute that learned computation into opaque KV states when the scratchpad is taken away.

**Phase 1 — Baseline**: Train a small transformer on long multiplication with correct, legible CoT supervision. The model learns the standard step-by-step algorithm.

**Phase 2 — Silent CoT**: Same architecture, same task, same CoT *length*, but CoT targets are replaced with zero/random tokens. Final correct answer is supervised. The model must learn to repurpose the forward passes through empty tokens as implicit computation steps.

we comapre these models on accuracy (with same parameter count and training batch):  

1: no COT model  
2: correct COT model  
3: Silent COT model that is curriculum learnt FROM a correct COT model  
4: Silent COT model trained directly  
5: Silent COT model trained from correct COT model with layer addition and gradient freeze *

so for 5, we first train a correct COT model with less layers than other models, then we add a couple of layers, and train on zero/random CoT. with maybe a lower LR on previous layers. the idea is we preserve the weight information on previous layers that carries the correct kv cache information better without making them all just predict 0.  

## Key Measurements

- **Accuracy drop**: Compare phase 1 vs phase 2 final answer accuracy
- **CoT ablation**: Randomize KV cache entries from CoT positions at inference. If accuracy drops, the model was using those representations for computation
- **Attention analysis**: Check whether final-answer tokens attend heavily to CoT-position KV states
- **Scaling with CoT length**: Does giving the model more empty tokens improve accuracy? If so, it's using them as compute


## mech interp

we attempt to find number features using mech interp on inner activations.  

## Expected Results

If phase 2 accuracy significantly exceeds what's achievable with zero CoT tokens (direct input → answer), the model has learned to exploit meaningless forward passes as working memory — silent reasoning.

## Implications

CoT faithfulness research assumes that monitoring the *text* of chain-of-thought is sufficient for interpretability. This work tests whether models can circumvent that entirely by computing through KV cache states that leave no legible trace in the token stream.
