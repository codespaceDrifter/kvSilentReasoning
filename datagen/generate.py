"""Generate long multiplication training data in 3 formats as tokenized binary.

Usage: python -m datagen.generate
Output: data/{correct_cot,silent_cot,no_cot}_{train,test}.bin

Formats:
  correct_cot: <bos> A * B = partial_products = result <eos>
  silent_cot:  <bos> A * B = _ _ _ ... _ = result <eos>  (same length, CoT nulled)
  no_cot:      <bos> A * B = result <eos>

Streams data to disk in chunks to avoid RAM issues.
"""

import random
import numpy as np
from pathlib import Path

from tokenizer.tokenizer import MulTokenizer

MAX_DIGITS = 5
# 72 tokens * 2 bytes = 144 bytes per example
BYTES_PER_EXAMPLE = 72 * 2
# target sizes
TRAIN_GB = 8
TEST_GB = 2
CHUNK_SIZE = 100_000


def num_to_tokens(n):
    """Convert integer to space-separated digit tokens. e.g. 123 -> '1 2 3'"""
    return ' '.join(str(n))


def mul_to_text(a, b):
    """Correct CoT: A * B = partial_products = result"""
    result = a * b
    a_tok = num_to_tokens(a)
    b_tok = num_to_tokens(b)

    b_digits = [int(d) for d in str(b)]
    # reverse: index 0 = ones place, index 1 = tens place, etc.
    b_digits_rev = b_digits[::-1]

    if len(b_digits) == 1:
        return f"{a_tok} * {b_tok} = {num_to_tokens(result)}"

    # partial products: a * (digit * 10^place) for each digit of b
    partials = [a * d * (10 ** place) for place, d in enumerate(b_digits_rev)]
    partials_tok = ' + '.join(num_to_tokens(p) for p in partials)

    return f"{a_tok} * {b_tok} = {partials_tok} = {num_to_tokens(result)}"


def mul_to_text_silent(a, b):
    """Silent CoT: same structure as correct but intermediate tokens replaced with '_'."""
    result = a * b
    b_digits = [int(d) for d in str(b)]

    if len(b_digits) == 1:
        # no CoT to null out, same as correct
        return f"{num_to_tokens(a)} * {num_to_tokens(b)} = {num_to_tokens(result)}"

    correct = mul_to_text(a, b)
    tokens = correct.split()

    # find first and last '=' positions
    eq_positions = [i for i, t in enumerate(tokens) if t == '=']
    first_eq, last_eq = eq_positions[0], eq_positions[-1]

    # replace everything between first '=' and last '=' with '_'
    for i in range(first_eq + 1, last_eq):
        tokens[i] = '_'

    return ' '.join(tokens)


def mul_to_text_no_cot(a, b):
    """No CoT: just A * B = result"""
    return f"{num_to_tokens(a)} * {num_to_tokens(b)} = {num_to_tokens(a * b)}"


# format name -> text generation function
FORMAT_FNS = {
    'correct_cot': mul_to_text,
    'silent_cot': mul_to_text_silent,
    'no_cot': mul_to_text_no_cot,
}


def random_pair(d1, d2, rng):
    """Generate random (a, b) where a has d1 digits and b has d2 digits."""
    lo_a, hi_a = 10 ** (d1 - 1), 10 ** d1 - 1
    lo_b, hi_b = 10 ** (d2 - 1), 10 ** d2 - 1
    return rng.randint(lo_a, hi_a), rng.randint(lo_b, hi_b)


def stream_generate(out_path, n_examples, format_fn, tokenizer, rng):
    """Generate n_examples using format_fn and stream to binary file."""
    generated = 0

    with open(out_path, 'wb') as f:
        while generated < n_examples:
            chunk_target = min(CHUNK_SIZE, n_examples - generated)
            # (chunk_target, seq_len)
            data = np.zeros((chunk_target, tokenizer.MAX_SEQ_LEN), dtype=np.int16)

            for idx in range(chunk_target):
                # uniform random difficulty level
                d1 = rng.randint(1, MAX_DIGITS)
                d2 = rng.randint(1, MAX_DIGITS)
                a, b = random_pair(d1, d2, rng)
                text = format_fn(a, b)
                ids = tokenizer.encode(text, pad=True)
                assert len(ids) == tokenizer.MAX_SEQ_LEN, \
                    f"seq len {len(ids)} != {tokenizer.MAX_SEQ_LEN}: {a} * {b} ({format_fn.__name__})"
                data[idx] = ids

            data.tofile(f)
            generated += chunk_target
            print(f'  {generated:,}/{n_examples:,} ({generated / n_examples * 100:.0f}%)')

    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f'wrote {out_path} ({size_gb:.2f} GB)')


def main():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()

    n_train = int(TRAIN_GB * (1024 ** 3) / BYTES_PER_EXAMPLE)
    n_test = int(TEST_GB * (1024 ** 3) / BYTES_PER_EXAMPLE)

    print(f'generating 6 splits:')
    print(f'  per format: {n_train:,} train ({TRAIN_GB} GB) + {n_test:,} test ({TEST_GB} GB)')
    print(f'  total: {3 * (n_train + n_test):,} examples ({3 * (TRAIN_GB + TEST_GB)} GB)')

    # different seed per split so problems don't overlap
    splits = [
        ('correct_cot', 'train', n_train, 100),
        ('correct_cot', 'test',  n_test,  200),
        ('silent_cot',  'train', n_train, 300),
        ('silent_cot',  'test',  n_test,  400),
        ('no_cot',      'train', n_train, 500),
        ('no_cot',      'test',  n_test,  600),
    ]

    for fmt, split, n, seed in splits:
        name = f'{fmt}_{split}'
        print(f'\ngenerating {name}...')
        rng = random.Random(seed)
        stream_generate(data_dir / f'{name}.bin', n, FORMAT_FNS[fmt], tokenizer, rng)

    print('\ndone!')


if __name__ == '__main__':
    main()
