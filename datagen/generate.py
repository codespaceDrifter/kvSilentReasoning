"""Generate long multiplication training data in 3 formats as tokenized binary.

Usage: python -m datagen.generate
Output: data/{correct_cot,silent_cot,no_cot}_{train,test}.bin

Same (a,b) pairs across all 3 formats. Test set is exactly 8192 pairs,
generated first with a separate RNG seed. Train set fills the rest.

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

MAX_DIGITS = 4
SEQ_LEN = 128
BYTES_PER_EXAMPLE = SEQ_LEN * 2
# train size in GB (per format file; total on disk = TRAIN_GB * 3 + test)
TRAIN_GB = 10
NUM_TEST = 8192
CHUNK_SIZE = 100_000

FORMATS = ['correct_cot', 'silent_cot', 'no_cot']


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

    eq_positions = [i for i, t in enumerate(tokens) if t == '=']
    first_eq, last_eq = eq_positions[0], eq_positions[-1]

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


def generate_split(n_pairs, rng, tokenizer, files, label):
    """Generate n_pairs and write all 3 formats to the corresponding files."""
    generated = 0
    while generated < n_pairs:
        chunk_size = min(CHUNK_SIZE, n_pairs - generated)

        # buffers per format
        buffers = {fmt: [] for fmt in FORMATS}

        for _ in range(chunk_size):
            d1 = rng.randint(1, MAX_DIGITS)
            d2 = rng.randint(1, MAX_DIGITS)
            a, b = random_pair(d1, d2, rng)

            for fmt, fn in FORMAT_FNS.items():
                text = fn(a, b)
                ids = tokenizer.encode(text, pad=True)
                assert len(ids) == SEQ_LEN, \
                    f"seq len {len(ids)} != {SEQ_LEN}: {a} * {b} ({fmt})"
                buffers[fmt].append(ids)

        for fmt, rows in buffers.items():
            arr = np.array(rows, dtype=np.int16)
            arr.tofile(files[fmt])

        generated += chunk_size
        if generated % (CHUNK_SIZE * 10) == 0 or generated >= n_pairs:
            pct = generated / n_pairs * 100
            print(f'  {label}: {generated:,}/{n_pairs:,} ({pct:.0f}%)')


def main():
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()

    n_train = int(TRAIN_GB * (1024 ** 3) / BYTES_PER_EXAMPLE)

    print(f'test:  {NUM_TEST:,} pairs (same across all 3 formats)')
    print(f'train: {n_train:,} pairs ({TRAIN_GB} GB per format)')

    # --- test: 8192 pairs, seed 0 ---
    test_files = {fmt: open(data_dir / f'{fmt}_test.bin', 'wb') for fmt in FORMATS}
    generate_split(NUM_TEST, random.Random(0), tokenizer, test_files, 'test')
    for f in test_files.values():
        f.close()

    # --- train: fill the rest, seed 42 ---
    train_files = {fmt: open(data_dir / f'{fmt}_train.bin', 'wb') for fmt in FORMATS}
    generate_split(n_train, random.Random(42), tokenizer, train_files, 'train')
    for f in train_files.values():
        f.close()

    print(f'\ndone! file sizes:')
    for split in ('test', 'train'):
        for fmt in FORMATS:
            path = data_dir / f'{fmt}_{split}.bin'
            size_gb = path.stat().st_size / (1024 ** 3)
            print(f'  {fmt}_{split}: {size_gb:.2f} GB')


if __name__ == '__main__':
    main()
