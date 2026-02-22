class MulTokenizer:
    TOKENS = [
        '<pad>', '<bos>', '<eos>',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '*', '+', '=',
        '_',
        'R',
    ]
    MAX_SEQ_LEN = 128
    # index of '=' token for loss masking
    EQUALS_ID = 15

    def __init__(self):
        self.tok2id = {t: i for i, t in enumerate(self.TOKENS)}
        self.id2tok = {i: t for i, t in enumerate(self.TOKENS)}
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text, pad=False):
        ids = [self.bos_id] + [self.tok2id[t] for t in text.split()] + [self.eos_id]
        if pad:
            ids += [self.pad_id] * (self.MAX_SEQ_LEN - len(ids))
        return ids

    def decode(self, ids):
        return ' '.join(self.id2tok[i] for i in ids if i not in (0, 1, 2))

    def __len__(self):
        return len(self.TOKENS)
