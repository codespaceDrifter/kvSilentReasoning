"""Silent Reasoning NAS.

Training tree (BRANCH_EPOCHS=B, TOTAL_EPOCHS=T):
  1. Train correct_cot for B epochs -> save checkpoint
  2. Continue correct_cot for T-B more epochs -> test = score 1
  3. Load checkpoint from step 1 -> train silent_cot for B epochs -> test = score 2
  4. Fresh weights -> train silent_cot for T epochs -> test = score 3
  5. Fresh weights -> train no_cot for T epochs -> test = score 4

Search space: layers {2,3,4} x heads {2,4,8} x head_dim {16,32,64} = 27 configs

Usage: python -m NAS.silent_reasoning
"""

import gc
import json
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model.gpt import GPT
from model.dataset import BinDataset
from tokenizer.tokenizer import MulTokenizer

# --- search space ---
LAYERS = [2, 3, 4]
HEADS = [2, 4, 8]
HEAD_DIMS = [16, 32, 64]

# --- training ---
MAX_SEQ_LEN = 128
DROPOUT = 0.1
BATCH_SIZE = 2048
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_FP16 = True

# epoch counts for the training tree
BRANCH_EPOCHS = 1
TOTAL_EPOCHS = 1

NUM_TEST = 16384

# --- paths ---
DATA_DIR = Path('data')
RESULTS_DIR = Path('NAS_results')
WEIGHTS_DIR = Path('weights')
LOG_FILE = RESULTS_DIR / 'log.json'
PLOT_FILE = RESULTS_DIR / 'plot.png'

# --- data split names ---
TRAIN_SPLITS = ['correct_cot_train', 'silent_cot_train', 'no_cot_train']
TEST_SPLITS = ['correct_cot_test', 'silent_cot_test', 'no_cot_test']


def get_configs():
    return [
        {'layers': l, 'num_heads': h, 'head_dim': hd, 'embed_dim': h * hd}
        for l in LAYERS for h in HEADS for hd in HEAD_DIMS
    ]


def predict_tokens(model, tokenizer, prefix_ids, max_gen=120):
    """Autoregressively generate from prefix until <eos> or max tokens."""
    # (1, prefix_len)
    input_ids = torch.tensor([prefix_ids], device='cuda')
    max_gen = min(max_gen, MAX_SEQ_LEN - len(prefix_ids))

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_FP16):
        for _ in range(max_gen):
            # (1, seq_len, vocab)
            logits = model(input_ids)
            next_id = logits[0, -1, :].argmax().item()
            # (1, seq_len+1)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device='cuda')], dim=1)
            if next_id == tokenizer.eos_id:
                break

    generated_ids = input_ids[0, len(prefix_ids):].tolist()
    return [tokenizer.id2tok[i] for i in generated_ids
            if i not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]


def extract_final_answer(tokens):
    """Extract tokens after the last '=' in the token list."""
    eq_positions = [i for i, t in enumerate(tokens) if t == '=']
    if eq_positions:
        return tokens[eq_positions[-1] + 1:]
    return tokens


def test_model(model, tokenizer, test_dataset, num_test=NUM_TEST):
    """Test model accuracy via autoregressive generation.

    Returns dict with 'complete' (full sequence match) and 'answer' (final answer match).
    """
    model.eval()
    equals_id = tokenizer.tok2id['=']
    rng = random.Random(42)
    n = min(num_test, len(test_dataset))
    indices = rng.sample(range(len(test_dataset)), n)

    complete_match = 0
    answer_match = 0

    for idx in indices:
        ids = test_dataset[idx].tolist()

        try:
            first_eq = ids.index(equals_id)
        except ValueError:
            continue

        # prefix: everything up to and including first '='
        prefix_ids = ids[:first_eq + 1]

        # expected: everything after first '=', excluding specials
        expected_tokens = [tokenizer.id2tok[x] for x in ids[first_eq + 1:]
                           if x not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]

        pred_tokens = predict_tokens(model, tokenizer, prefix_ids)

        if pred_tokens == expected_tokens:
            complete_match += 1

        if extract_final_answer(pred_tokens) == extract_final_answer(expected_tokens):
            answer_match += 1

    return {
        'complete': round(complete_match / n * 100, 1),
        'answer': round(answer_match / n * 100, 1),
    }


def train_epochs(model, loader, equals_id, num_epochs, lr=LR):
    """Train model for num_epochs over the dataloader. Returns average loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_FP16)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in loader:
            batch = batch.cuda()

            with torch.amp.autocast('cuda', enabled=USE_FP16):
                loss = model.compute_loss(batch, equals_id)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_batches += 1

        avg_epoch_loss = epoch_loss / epoch_batches
        total_loss += epoch_loss
        num_batches += epoch_batches
        print(f'      epoch {epoch + 1}/{num_epochs}, loss {avg_epoch_loss:.4f}')

    return total_loss / num_batches


def run_config(config, tokenizer, loaders, test_datasets, equals_id):
    """Train and evaluate all 4 model variants for one architecture config."""
    l = config['layers']
    h = config['num_heads']
    hd = config['head_dim']
    d = config['embed_dim']
    name = f'L{l}_H{h}_HD{hd}'

    print(f'\n{"="*60}')
    print(f'Config: {name} (d_model={d})')
    print(f'{"="*60}')

    def make_model():
        return GPT(
            vocab_size=len(tokenizer),
            d_model=d,
            n_heads=h,
            n_layers=l,
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT,
            pad_id=tokenizer.pad_id,
        ).cuda()

    result = {**config}
    tmp = make_model()
    result['params'] = sum(p.numel() for p in tmp.parameters())
    del tmp; gc.collect(); torch.cuda.empty_cache()
    print(f'params: {result["params"]:,}')

    # === Correct CoT: 4 epochs -> checkpoint -> 4 more epochs ===
    print(f'\n  [1/4] correct_cot ({TOTAL_EPOCHS} epochs, branch at {BRANCH_EPOCHS})...')
    model = make_model()
    t0 = time.time()

    print(f'    phase 1: {BRANCH_EPOCHS} epochs...')
    loss1 = train_epochs(model, loaders['correct_cot_train'], equals_id, BRANCH_EPOCHS)

    # save intermediate checkpoint for curriculum branch
    branch_state = {k: v.clone() for k, v in model.state_dict().items()}

    remaining = TOTAL_EPOCHS - BRANCH_EPOCHS
    if remaining > 0:
        print(f'    phase 2: {remaining} more epochs...')
        loss2 = train_epochs(model, loaders['correct_cot_train'], equals_id, remaining)
        loss = (loss1 + loss2) / 2
    else:
        loss = loss1
    train_time = time.time() - t0

    print(f'  testing correct_cot...')
    acc = test_model(model, tokenizer, test_datasets['correct_cot_test'])
    print(f'  correct_cot: complete={acc["complete"]}% answer={acc["answer"]}% '
          f'loss={loss:.4f} time={train_time:.0f}s')

    torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_correct_cot.pt')
    result['correct_cot'] = {**acc, 'loss': round(loss, 4), 'train_time': round(train_time, 1)}
    del model; gc.collect(); torch.cuda.empty_cache()

    # === Silent CoT Curriculum: load branch checkpoint -> 4 epochs ===
    print(f'\n  [2/4] silent_cot_curriculum ({BRANCH_EPOCHS} epochs from branch)...')
    model = make_model()
    model.load_state_dict(branch_state)
    del branch_state
    t0 = time.time()
    loss = train_epochs(model, loaders['silent_cot_train'], equals_id, BRANCH_EPOCHS)
    train_time = time.time() - t0

    print(f'  testing silent_cot_curriculum...')
    acc = test_model(model, tokenizer, test_datasets['silent_cot_test'])
    print(f'  silent_cot_curriculum: complete={acc["complete"]}% answer={acc["answer"]}% '
          f'loss={loss:.4f} time={train_time:.0f}s')

    torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_silent_cot_curriculum.pt')
    result['silent_cot_curriculum'] = {**acc, 'loss': round(loss, 4), 'train_time': round(train_time, 1)}
    del model; gc.collect(); torch.cuda.empty_cache()

    # === Silent CoT Scratch: fresh -> 8 epochs ===
    print(f'\n  [3/4] silent_cot_scratch ({TOTAL_EPOCHS} epochs)...')
    model = make_model()
    t0 = time.time()
    loss = train_epochs(model, loaders['silent_cot_train'], equals_id, TOTAL_EPOCHS)
    train_time = time.time() - t0

    print(f'  testing silent_cot_scratch...')
    acc = test_model(model, tokenizer, test_datasets['silent_cot_test'])
    print(f'  silent_cot_scratch: complete={acc["complete"]}% answer={acc["answer"]}% '
          f'loss={loss:.4f} time={train_time:.0f}s')

    torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_silent_cot_scratch.pt')
    result['silent_cot_scratch'] = {**acc, 'loss': round(loss, 4), 'train_time': round(train_time, 1)}
    del model; gc.collect(); torch.cuda.empty_cache()

    # === No CoT: fresh -> 8 epochs ===
    print(f'\n  [4/4] no_cot ({TOTAL_EPOCHS} epochs)...')
    model = make_model()
    t0 = time.time()
    loss = train_epochs(model, loaders['no_cot_train'], equals_id, TOTAL_EPOCHS)
    train_time = time.time() - t0

    print(f'  testing no_cot...')
    acc = test_model(model, tokenizer, test_datasets['no_cot_test'])
    print(f'  no_cot: complete={acc["complete"]}% answer={acc["answer"]}% '
          f'loss={loss:.4f} time={train_time:.0f}s')

    torch.save(model.state_dict(), WEIGHTS_DIR / f'{name}_no_cot.pt')
    result['no_cot'] = {**acc, 'loss': round(loss, 4), 'train_time': round(train_time, 1)}
    del model; gc.collect(); torch.cuda.empty_cache()

    return result


def plot_results(results):
    """Grouped bar chart: 4 accuracy scores per config, subplots by layer count."""
    score_types = ['correct_cot', 'silent_cot_curriculum', 'silent_cot_scratch', 'no_cot']
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    labels = ['Correct CoT', 'Silent CoT (curriculum)', 'Silent CoT (scratch)', 'No CoT']

    fig, axes = plt.subplots(1, len(LAYERS), figsize=(7 * len(LAYERS), 6), sharey=True)
    if len(LAYERS) == 1:
        axes = [axes]

    for ax, n_layers in zip(axes, LAYERS):
        layer_results = sorted(
            [r for r in results if r['layers'] == n_layers],
            key=lambda r: (r['num_heads'], r['head_dim']),
        )
        if not layer_results:
            continue

        x_labels = [f"H={r['num_heads']}\nHD={r['head_dim']}" for r in layer_results]
        x = np.arange(len(layer_results))
        width = 0.18

        for i, (score_type, color, label) in enumerate(zip(score_types, colors, labels)):
            values = [r.get(score_type, {}).get('answer', 0) for r in layer_results]
            ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)

        ax.set_xlabel('Architecture')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(f'{n_layers} Layers')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Answer Accuracy (%)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    plt.suptitle('Silent Reasoning NAS: Answer Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved plot: {PLOT_FILE}')


def main():
    random.seed(42)
    torch.manual_seed(42)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()
    equals_id = tokenizer.tok2id['=']

    configs = get_configs()
    print(f'Silent Reasoning NAS: {len(configs)} configs')
    for c in configs:
        print(f"  L={c['layers']} H={c['num_heads']} HD={c['head_dim']} (d_model={c['embed_dim']})")

    # --- load all datasets (mmap, instant) ---
    print('\nloading datasets...')
    all_splits = TRAIN_SPLITS + TEST_SPLITS
    datasets = {}
    for split_name in all_splits:
        path = DATA_DIR / f'{split_name}.bin'
        assert path.exists(), f'missing data file: {path} — run `python -m datagen.generate` first'
        datasets[split_name] = BinDataset(str(path), seq_len=MAX_SEQ_LEN)
        print(f'  {split_name}: {len(datasets[split_name]):,} examples')

    # train dataloaders (shuffle for epoch-based training)
    loaders = {
        split_name: DataLoader(datasets[split_name], batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
        for split_name in TRAIN_SPLITS
    }

    # test datasets (accessed by index for autoregressive eval)
    test_datasets = {split_name: datasets[split_name] for split_name in TEST_SPLITS}

    # --- load existing results for resuming ---
    results = []
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            results = json.load(f)
        print(f'\nloaded {len(results)} existing results')

    done_keys = {(r['layers'], r['num_heads'], r['head_dim']) for r in results}

    # --- run all configs ---
    for i, config in enumerate(configs):
        key = (config['layers'], config['num_heads'], config['head_dim'])
        if key in done_keys:
            print(f'\nskipping [{i+1}/{len(configs)}] L={key[0]} H={key[1]} HD={key[2]} (done)')
            continue

        print(f'\n[{i+1}/{len(configs)}]')
        result = run_config(config, tokenizer, loaders, test_datasets, equals_id)
        results.append(result)

        # save after each config (resume-safe)
        with open(LOG_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        plot_results(results)

    # --- summary ---
    print('\n' + '=' * 60)
    print('SILENT REASONING NAS COMPLETE')
    print(f'results: {LOG_FILE}')
    print(f'plot: {PLOT_FILE}')
    print('=' * 60)

    print('\nTop 5 by correct_cot answer accuracy:')
    by_acc = sorted(results, key=lambda r: r['correct_cot']['answer'], reverse=True)
    for r in by_acc[:5]:
        print(f"  L={r['layers']} H={r['num_heads']} HD={r['head_dim']}: "
              f"correct={r['correct_cot']['answer']}% "
              f"curriculum={r['silent_cot_curriculum']['answer']}% "
              f"scratch={r['silent_cot_scratch']['answer']}% "
              f"no_cot={r['no_cot']['answer']}%")

    print('\nTop 5 by silent_cot_curriculum answer accuracy:')
    by_acc = sorted(results, key=lambda r: r['silent_cot_curriculum']['answer'], reverse=True)
    for r in by_acc[:5]:
        print(f"  L={r['layers']} H={r['num_heads']} HD={r['head_dim']}: "
              f"curriculum={r['silent_cot_curriculum']['answer']}% "
              f"(correct={r['correct_cot']['answer']}%)")


if __name__ == '__main__':
    main()
