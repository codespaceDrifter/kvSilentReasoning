"""Dataset for loading pre-tokenized data from binary files."""

import numpy as np
import torch
from torch.utils.data import Dataset


class BinDataset(Dataset):
    """Load pre-tokenized data from binary file."""

    def __init__(self, bin_path: str, seq_len: int = 128):
        # load as memory-mapped array for efficiency with large files
        self.data = np.memmap(bin_path, dtype=np.int16, mode='r')
        # (num_examples, seq_len)
        self.data = self.data.reshape(-1, seq_len)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # (seq_len,) int16 -> long tensor
        return torch.from_numpy(self.data[idx].astype(np.int64))
