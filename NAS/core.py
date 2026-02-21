"""Shared NAS functionality."""

import gc
import json
import random
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass

from model.gpt import GPT
from model.dataset import BinDataset
from tokenizer.tokenizer import MulTokenizer
from datagen.generate import mul_to_text, random_pair


@dataclass
class NASConfig:
    """Config for a NAS run."""
    name: str
    configs: list
    max_batches: int = 100_000
    batch_size: int = 8192
    num_test: int = 5_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    use_fp16: bool = True
    max_seq_len: int = 72


def get_paths(name):
    results_dir = Path('NAS_results') / name
    weights_dir = results_dir / 'weights'
    log_file = results_dir / 'log.json'
    return results_dir, weights_dir, log_file


def predict_answer(model, tokenizer, prefix_ids, use_fp16=True, max_gen=60):
    """Given prefix (up to and including first '='), predict the rest."""
    # (1, prefix_len)
    input_ids = torch.tensor([prefix_ids], device='cuda')

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_fp16):
        for _ in range(max_gen):
            # (1, seq_len, vocab)
            logits = model(input_ids)
            next_id = logits[0, -1, :].argmax().item()
            # (1, seq_len+1)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device='cuda')], dim=1)
            if next_id == tokenizer.eos_id:
                break

    generated_ids = input_ids[0, len(prefix_ids):].tolist()
    return [tokenizer.id2tok[i] for i in generated_ids if i not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]


def test_accuracy(model, tokenizer, num_test, use_fp16=True):
    """Test model accuracy on random multiplication problems."""
    model.eval()
    num_correct = 0
    rng = random.Random(123)
    equals_id = tokenizer.tok2id['=']

    with torch.amp.autocast('cuda', enabled=use_fp16):
        for _ in range(num_test):
            # random difficulty
            d1 = rng.randint(1, 5)
            d2 = rng.randint(1, 5)
            a, b = random_pair(d1, d2, rng)

            text = mul_to_text(a, b)
            full_ids = tokenizer.encode(text, pad=False)

            # prefix up to and including first '='
            eq_pos = full_ids.index(equals_id)
            prefix_ids = full_ids[:eq_pos + 1]

            # true answer tokens (after first '=')
            true_ids = full_ids[eq_pos + 1:]
            true_tokens = [tokenizer.id2tok[i] for i in true_ids if i not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]

            pred_tokens = predict_answer(model, tokenizer, prefix_ids, use_fp16)

            if pred_tokens == true_tokens:
                num_correct += 1

    return num_correct / num_test * 100


def train_batches(model, train_loader, equals_id, cfg):
    """Train model for max_batches. Returns average loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_fp16)

    total_loss = 0.0
    num_batches = 0
    train_iter = iter(train_loader)
    log_interval = max(cfg.max_batches // 10, 1000)

    while num_batches < cfg.max_batches:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.cuda()

        with torch.amp.autocast('cuda', enabled=cfg.use_fp16):
            loss = model.compute_loss(batch, equals_id)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % log_interval == 0:
            print(f'    batch {num_batches}/{cfg.max_batches}, loss {loss.item():.4f}')

    return total_loss / num_batches


def run_config(config, tokenizer, train_loader, equals_id, cfg, weights_dir):
    """Train and test one config. Returns result dict."""
    n_layers = config['layers']
    d_model = config['embed_dim']
    n_heads = config['num_heads']
    head_dim = d_model // n_heads

    print(f'\n{"="*60}')
    print(f'Config: layers={n_layers}, embed_dim={d_model}, num_heads={n_heads}, head_dim={head_dim}')
    print(f'{"="*60}')

    model = GPT(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        pad_id=tokenizer.pad_id,
    )
    model = model.cuda()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'params: {num_params:,}')

    start_time = time.time()
    avg_loss = train_batches(model, train_loader, equals_id, cfg)
    train_time = time.time() - start_time
    print(f'{cfg.max_batches} batches in {train_time:.1f}s, avg_loss: {avg_loss:.4f}')

    print(f'testing on {cfg.num_test} random examples...')
    accuracy = test_accuracy(model, tokenizer, cfg.num_test, cfg.use_fp16)
    print(f'accuracy: {accuracy:.1f}%')

    weight_name = f'L{n_layers}_E{d_model}_H{n_heads}.pt'
    weight_path = weights_dir / weight_name
    torch.save(model.state_dict(), weight_path)
    print(f'saved: {weight_path}')

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'layers': n_layers,
        'embed_dim': d_model,
        'num_heads': n_heads,
        'head_dim': head_dim,
        'params': num_params,
        'loss': round(avg_loss, 4),
        'accuracy': round(accuracy, 1),
        'train_time': round(train_time, 1),
    }


def run_nas(cfg):
    """Run full NAS with given config."""
    random.seed(42)
    torch.manual_seed(42)

    results_dir, weights_dir, log_file = get_paths(cfg.name)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    tokenizer = MulTokenizer()
    equals_id = tokenizer.tok2id['=']

    print(f'NAS: {cfg.name}')
    print(f'configs: {len(cfg.configs)}')
    print(f'max_batches: {cfg.max_batches}, batch_size: {cfg.batch_size}')

    print('\nloading training data...')
    data_dir = Path('data')
    train_dataset = BinDataset(str(data_dir / 'train.bin'), seq_len=cfg.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f'train examples: {len(train_dataset)}')

    results = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            results = json.load(f)
        print(f'loaded {len(results)} existing results')

    done_keys = {(r['layers'], r['embed_dim'], r['num_heads']) for r in results}

    for i, config in enumerate(cfg.configs):
        key = (config['layers'], config['embed_dim'], config['num_heads'])
        if key in done_keys:
            print(f'\nskipping {i+1}/{len(cfg.configs)} (already done)')
            continue

        print(f'\n[{i+1}/{len(cfg.configs)}]')
        result = run_config(config, tokenizer, train_loader, equals_id, cfg, weights_dir)
        results.append(result)

        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '='*60)
    print(f'{cfg.name.upper()} NAS COMPLETE')
    print(f'results: {log_file}')
    print('='*60)

    by_acc = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    print('\nTop 5 by accuracy:')
    for r in by_acc[:5]:
        print(f"  L={r['layers']} E={r['embed_dim']} H={r['num_heads']}: {r['accuracy']}% ({r['params']:,} params)")

    return results
