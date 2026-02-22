"""Generate long multiplication training data in 3 formats as tokenized binary.

Usage: python -m datagen.generate
Output: data/{correct_cot,silent_cot,no_cot}_{train,test}.bin

Each (a,b) pair is assigned to exactly ONE of 6 splits (no overlap).
Random bucket assignment with weights proportional to desired file sizes.

Digit order in outputs is REVERSED (least significant first) so that carry
propagation flows in the generation direction. An 'R' token marks reversed
sections. The final answer after the last '=' is in normal (MSB-first) order.

Formats:
  correct_cot: <bos> A * B = R rev_partials = R rev_sum = answer <eos>
  silent_cot:  <bos> A * B = _ _ _ ... _ = answer <eos>  (same length)
  no_cot:      <bos> A * B = answer <eos>
"""

import random
import numpy as np
from pathlib import Path

from tokenizer.tokenizer import MulTokenizer

MAX_DIGITS = 5
SEQ_LEN = 128
BYTES_PER_EXAMPLE = SEQ_LEN * 2
TOTAL_GB = 30
CHUNK_SIZE = 100_000


def num_to_tokens(n):
    """Convert integer to space-separated digit tokens (normal order). e.g. 123 -> '1 2 3'"""
    return ' '.join(str(n))


def num_to_tokens_rev(n):
    """Convert integer to space-separated digit tokens (reversed). e.g. 123 -> '3 2 1'"""
    return ' '.join(str(n)[::-1])


def mul_to_text(a, b):
    """Correct CoT with reversed intermediate digits.

    Format: A * B = R rev_partial1 + rev_partial2 + ... = R rev_sum = answer
    Single-digit b (no partials to decompose): A * B = R rev_answer = answer
    """
    result = a * b
    a_tok = num_to_tokens(a)
    b_tok = num_to_tokens(b)

    b_digits = [int(d) for d in str(b)]
    # reverse: index 0 = ones place, index 1 = tens place, etc.
    b_digits_rev = b_digits[::-1]

    if len(b_digits) == 1:
        # no partial products to show, just reversed result then normal result
        return f"{a_tok} * {b_tok} = R {num_to_tokens_rev(result)} = {num_to_tokens(result)}"

    # partial products: a * (digit * 10^place) for each digit of b
    partials = [a * d * (10 ** place) for place, d in enumerate(b_digits_rev)]
    partials_tok = ' + '.join(num_to_tokens_rev(p) for p in partials)

    return (f"{a_tok} * {b_tok} = R {partials_tok} = "
            f"R {num_to_tokens_rev(result)} = {num_to_tokens(result)}")


def mul_to_text_silent(a, b):
    """Silent CoT: same length as correct_cot but intermediate tokens replaced with '_'.

    Only the final normal-order answer (after last '=') is visible.
    """
    correct = mul_to_text(a, b)
    tokens = correct.split()

    # find all '=' positions
    eq_positions = [i for i, t in enumerate(tokens) if t == '=']
    first_eq, last_eq = eq_positions[0], eq_positions[-1]

    # replace everything between first '=' and last '=' with '_'
    for i in range(first_eq + 1, last_eq):
        tokens[i] = '_'

    return ' '.join(tokens)


def mul_to_text_no_cot(a, b):
    """No CoT: just A * B = answer (normal digit order)."""
    return f"{num_to_tokens(a)} * {num_to_tokens(b)} = {num_to_tokens(a * b)}"


def random_pair(d1, d2, rng):
    """Generate random (a, b) where a has d1 digits and b has d2 digits."""
    lo_a, hi_a = 10 ** (d1 - 1), 10 ** d1 - 1
    lo_b, hi_b = 10 ** (d2 - 1), 10 ** d2 - 1
    return rng.randint(lo_a, hi_a), rng.randint(lo_b, hi_b)


FORMAT_FNS = {
    'correct_cot': mul_to_text,
    'silent_cot': mul_to_text_silent,
    'no_cot': mul_to_text_no_cot,
}

# 6 buckets: each (a,b) pair goes to exactly one
# weights: 4 train + 1 test per format = 5 per format, 3 formats = 15 total
BUCKETS = [
    ('correct_cot', 'train', 4),
    ('correct_cot', 'test',  1),
    ('silent_cot',  'train', 4),
    ('silent_cot',  'test',  1),
    ('no_cot',      'train', 4),
    ('no_cot',      'test',  1),
]
# bucket keys
BUCKET_KEYS = [f'{fmt}_{split}' for fmt, split, _ in BUCKETS]
# cumulative weights for np.searchsorted
BUCKET_WEIGHTS = np.array([w for _, _, w in BUCKETS], dtype=np.float64)
BUCKET_CDF = np.cumsum(BUCKET_WEIGHTS / BUCKET_WEIGHTS.sum())


def main():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()

    # total pairs to generate (each becomes 1 example in 1 of 6 files)
    n_total = int(TOTAL_GB * (1024 ** 3) / BYTES_PER_EXAMPLE)

    print(f'generating {n_total:,} examples across 6 splits (no overlap)')
    print(f'target: ~{TOTAL_GB} GB total')
    print(f'bucket weights: {", ".join(f"{k}={w}" for k, (_, _, w) in zip(BUCKET_KEYS, BUCKETS))}')

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    # open all 6 files simultaneously
    files = {k: open(data_dir / f'{k}.bin', 'wb') for k in BUCKET_KEYS}
    counts = {k: 0 for k in BUCKET_KEYS}

    generated = 0
    while generated < n_total:
        chunk_size = min(CHUNK_SIZE, n_total - generated)

        # assign each example to a bucket via CDF lookup
        rolls = np_rng.random(chunk_size)
        # bucket index per example
        bucket_ids = np.searchsorted(BUCKET_CDF, rolls)

        # buffers per bucket
        buffers = {k: [] for k in BUCKET_KEYS}

        for idx in range(chunk_size):
            d1 = rng.randint(1, MAX_DIGITS)
            d2 = rng.randint(1, MAX_DIGITS)
            a, b = random_pair(d1, d2, rng)

            bucket_key = BUCKET_KEYS[bucket_ids[idx]]
            # format is determined by which bucket this pair landed in
            fmt = bucket_key.rsplit('_', 1)[0]
            text = FORMAT_FNS[fmt](a, b)
            ids = tokenizer.encode(text, pad=True)
            assert len(ids) == SEQ_LEN, \
                f"seq len {len(ids)} != {SEQ_LEN}: {a} * {b} ({fmt})"
            buffers[bucket_key].append(ids)

        for key, rows in buffers.items():
            if rows:
                arr = np.array(rows, dtype=np.int16)
                arr.tofile(files[key])
                counts[key] += len(rows)

        generated += chunk_size
        if generated % (CHUNK_SIZE * 10) == 0 or generated >= n_total:
            pct = generated / n_total * 100
            print(f'  {generated:,}/{n_total:,} ({pct:.0f}%)')

    for f in files.values():
        f.close()

    print(f'\ndone! file sizes:')
    for key in sorted(counts):
        path = data_dir / f'{key}.bin'
        size_gb = path.stat().st_size / (1024 ** 3)
        print(f'  {key}: {counts[key]:,} examples ({size_gb:.2f} GB)')


if __name__ == '__main__':
    main()
