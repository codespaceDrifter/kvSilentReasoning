"""Unified NAS search.

Search space:
- layers: 2, 3
- embed_dim: 24, 28
- num_heads: 2, 4
Plus special: 1L/256E/16H

Usage: python -m NAS.nas
"""

from NAS.core import NASConfig, run_nas

# main search: 2x2x2 = 8 configs
LAYERS = [2, 3]
EMBED_DIMS = [24, 28]
NUM_HEADS = [2, 4]


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

    # special: wide 1-layer
    configs.append({
        'layers': 1,
        'embed_dim': 256,
        'num_heads': 16,
    })

    return configs


if __name__ == '__main__':
    cfg = NASConfig(
        name='nas',
        configs=get_configs(),
        max_batches=100_000,
        batch_size=6144,
    )
    print(f'configs: {len(cfg.configs)}')
    for c in cfg.configs:
        print(f"  L={c['layers']} E={c['embed_dim']} H={c['num_heads']}")
    run_nas(cfg)
