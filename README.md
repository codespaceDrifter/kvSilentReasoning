# Silent Reasoning: Can Transformers Use KV Cache of Null CoT to Reason?

We investigate whether transformers can mesa-optimize their KV cache to perform computation through meaningless chain-of-thought tokens. If a model can produce correct answers to multi-step reasoning tasks while its CoT consists of zero/random tokens, it implies the forward passes themselves — not the token content — carry the computation via learned KV cache representations.

This has direct safety implications: CoT monitoring becomes insufficient if models can "think" through opaque internal states rather than legible text.

## Method

The two-phase design is critical. A model trained from scratch with empty CoT and correct final answers has no algorithm to hide — it would just memorize or fail. By first teaching the model *how* to solve the task through legible reasoning, we then test whether it can reroute that learned computation into opaque KV states when the scratchpad is taken away.

**Phase 1 — Baseline**: Train a small transformer on long multiplication with correct, legible CoT supervision. The model learns the standard step-by-step algorithm.

**Phase 2 — Silent CoT**: Same architecture, same task, same CoT *length*, but CoT targets are replaced with `_` (null) tokens. Final correct answer is supervised. The model must learn to repurpose the forward passes through empty tokens as implicit computation steps.

We compare 4 model variants per architecture. With `BRANCH_EPOCHS=B` and `TOTAL_EPOCHS=T`:

```
correct_cot:  [B epochs correct] ──checkpoint──> [T-B more epochs correct] -> test
                                        │
curriculum:                             └──> [B epochs silent_cot] -> test

scratch:      [T epochs silent_cot from random init] -> test
no_cot:       [T epochs no_cot from random init] -> test
```

1. **Correct CoT** — T epochs on full chain-of-thought with partial products
2. **Silent CoT (curriculum)** — B epochs correct CoT, then B epochs null CoT (branches from correct CoT checkpoint)
3. **Silent CoT (scratch)** — T epochs on null CoT from random init
4. **No CoT** — T epochs on direct `A * B = answer`, no intermediate steps

## Reversed Digit Order and the R Token

### Why reversed?

Standard left-to-right digit order fights the computation. Consider generating `13 * 5 = 65`:
- The model must output `6` first (most significant digit)
- But to know it's `6`, it needs the carry from `3*5=15` (carry 1)
- So the model must compute the ENTIRE multiplication in a single forward pass just to output the first digit

This is because carry propagation flows right-to-left (least significant to most significant), but autoregressive generation flows left-to-right. Every output digit requires knowing all carries from less significant positions that haven't been generated yet.

### The fix

We reverse the digit order in all computed outputs. An `R` token marks reversed sections:

```
13 * 45:
  correct_cot: 1 3 * 4 5 = R 5 6 + 0 2 5 = R 5 8 5 = 5 8 5
  silent_cot:  1 3 * 4 5 = _ _ _ _ _ _ _ _ _ _ _ _ = 5 8 5
  no_cot:      1 3 * 4 5 = 5 8 5
```

Now generating the first output digit `5` (ones place of `13*5=65`):
- Just compute `(3 * 5) mod 10 = 5`. One multiplication, no carry needed.

Each subsequent digit only needs the carry from the previous step, which was already generated and is visible via attention. This makes each token O(1) computation instead of O(n).

The `R` token marks where reversed digit sequences begin. After the last `=`, the final answer is in normal (human-readable) order — the model just reads its reversed result backwards, which is a trivial attention pattern.

## NAS Search

Grid search over architectures: layers {2,3,4} x heads {2,4,8} x head_dim {16,32,64} = 27 configs. Each config trains all 4 model variants.

## Usage

```bash
python -m datagen.generate      # ~30GB data
python -m NAS.silent_reasoning   # 27 configs x 4 models
```

## Key Measurements

- **Accuracy drop**: Compare correct CoT vs silent CoT final answer accuracy
- **CoT ablation**: Randomize KV cache entries from CoT positions at inference. If accuracy drops, the model was using those representations for computation
- **Attention analysis**: Check whether final-answer tokens attend heavily to CoT-position KV states
- **Scaling with CoT length**: Does giving the model more empty tokens improve accuracy? If so, it's using them as compute

## Expected Results

If silent CoT accuracy significantly exceeds no-CoT accuracy, the model has learned to exploit meaningless forward passes as working memory — silent reasoning.

## Implications

CoT faithfulness research assumes that monitoring the *text* of chain-of-thought is sufficient for interpretability. This work tests whether models can circumvent that entirely by computing through KV cache states that leave no legible trace in the token stream.
