"""Pure Attention NAS (no MLP, no O projection).

Search: 2 layers, embed_dim 24/28/32, heads 2/4

Usage: python -m NAS.pure_attention
"""

import gc
import json
import random
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model.pure_attention import PureAttentionGPT
from model.dataset import BinDataset
from tokenizer.tokenizer import MulTokenizer
from datagen.generate import mul_to_text, random_pair

# search space
LAYERS = [2]
EMBED_DIMS = [24, 28, 32]
NUM_HEADS = [2, 4]

# training config
MAX_SEQ_LEN = 72
DROPOUT = 0.1
BATCH_SIZE = 6144
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_FP16 = True
MAX_BATCHES = 100_000
NUM_TEST = 5_000

# paths
DATA_DIR = Path('data')
RESULTS_DIR = Path('NAS_results/pure_attention')
WEIGHTS_DIR = RESULTS_DIR / 'weights'
LOG_FILE = RESULTS_DIR / 'log.json'


def get_configs():
    configs = []
    for n_layers in LAYERS:
        for d_model in EMBED_DIMS:
            for n_heads in NUM_HEADS:
                if d_model % n_heads == 0:
                    configs.append({
                        'layers': n_layers,
                        'embed_dim': d_model,
                        'num_heads': n_heads,
                    })
    return configs


def predict_answer(model, tokenizer, prefix_ids):
    input_ids = torch.tensor([prefix_ids], device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_FP16):
        for _ in range(60):
            logits = model(input_ids)
            next_id = logits[0, -1, :].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device='cuda')], dim=1)
            if next_id == tokenizer.eos_id:
                break
    generated_ids = input_ids[0, len(prefix_ids):].tolist()
    return [tokenizer.id2tok[i] for i in generated_ids if i not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]


def test_accuracy(model, tokenizer, num_test):
    model.eval()
    num_correct = 0
    rng = random.Random(123)
    equals_id = tokenizer.tok2id['=']

    with torch.amp.autocast('cuda', enabled=USE_FP16):
        for _ in range(num_test):
            d1 = rng.randint(1, 5)
            d2 = rng.randint(1, 5)
            a, b = random_pair(d1, d2, rng)
            text = mul_to_text(a, b)
            full_ids = tokenizer.encode(text, pad=False)
            eq_pos = full_ids.index(equals_id)
            prefix_ids = full_ids[:eq_pos + 1]
            true_ids = full_ids[eq_pos + 1:]
            true_tokens = [tokenizer.id2tok[i] for i in true_ids if i not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]
            if predict_answer(model, tokenizer, prefix_ids) == true_tokens:
                num_correct += 1
    return num_correct / num_test * 100


def train(model, train_loader, equals_id):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_FP16)
    total_loss = 0.0
    num_batches = 0
    train_iter = iter(train_loader)
    log_interval = MAX_BATCHES // 5

    while num_batches < MAX_BATCHES:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.cuda()
        with torch.amp.autocast('cuda', enabled=USE_FP16):
            loss = model.compute_loss(batch, equals_id)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
        if num_batches % log_interval == 0:
            print(f'    batch {num_batches}/{MAX_BATCHES}, loss {loss.item():.4f}')
    return total_loss / num_batches


def run_config(config, tokenizer, train_loader, equals_id):
    n_layers = config['layers']
    d_model = config['embed_dim']
    n_heads = config['num_heads']

    print(f'\n{"="*60}')
    print(f'Pure Attention: L={n_layers} E={d_model} H={n_heads}')
    print('='*60)

    model = PureAttentionGPT(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    model = model.cuda()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'params: {num_params:,}')

    start_time = time.time()
    avg_loss = train(model, train_loader, equals_id)
    train_time = time.time() - start_time
    print(f'{MAX_BATCHES} batches in {train_time:.1f}s, loss: {avg_loss:.4f}')

    print(f'testing {NUM_TEST} examples...')
    accuracy = test_accuracy(model, tokenizer, NUM_TEST)
    print(f'accuracy: {accuracy:.1f}%')

    weight_file = WEIGHTS_DIR / f'L{n_layers}_E{d_model}_H{n_heads}.pt'
    torch.save(model.state_dict(), weight_file)
    print(f'saved: {weight_file}')

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'layers': n_layers,
        'embed_dim': d_model,
        'num_heads': n_heads,
        'params': num_params,
        'loss': round(avg_loss, 4),
        'accuracy': round(accuracy, 1),
        'train_time': round(train_time, 1),
    }


def main():
    random.seed(42)
    torch.manual_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()
    equals_id = tokenizer.tok2id['=']

    configs = get_configs()
    print(f'Pure Attention NAS: {len(configs)} configs')
    for c in configs:
        print(f"  L={c['layers']} E={c['embed_dim']} H={c['num_heads']}")

    print('\nloading data...')
    train_dataset = BinDataset(str(DATA_DIR / 'train.bin'), seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f'train: {len(train_dataset)}')

    results = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            results = json.load(f)
    done = {(r['layers'], r['embed_dim'], r['num_heads']) for r in results}

    for config in configs:
        key = (config['layers'], config['embed_dim'], config['num_heads'])
        if key in done:
            print(f'\nskipping L={key[0]} E={key[1]} H={key[2]} (done)')
            continue
        result = run_config(config, tokenizer, train_loader, equals_id)
        results.append(result)
        with open(LOG_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '='*60)
    print('PURE ATTENTION NAS COMPLETE')
    print('='*60)
    for r in sorted(results, key=lambda x: -x['accuracy']):
        print(f"  L={r['layers']} E={r['embed_dim']} H={r['num_heads']}: {r['accuracy']}%")


if __name__ == '__main__':
    main()
