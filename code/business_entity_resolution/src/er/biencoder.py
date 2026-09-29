"""Multilingual bi-encoder (e5-style mean pooling) for candidate generation."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# set HF_HOME yourself to choose where the base model is cached
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = "intfloat/multilingual-e5-small"
MAX_LEN = 64


def make_text(name: str, addr: str) -> str:
    return f"query: {name} | {addr}"


def texts_of(df) -> list[str]:
    return [make_text(n, a) for n, a in zip(df["business_name"].to_list(), df["business_address"].to_list())]


def mean_pool(h, am):
    am = am.unsqueeze(-1).to(h.dtype)
    return (h * am).sum(1) / am.sum(1).clamp(min=1)


def load_model(path=BASE, train=False):
    tok = AutoTokenizer.from_pretrained(path)
    # train: fp32 master weights + autocast fp16; eval: pure fp16 (V100: never bf16)
    m = AutoModel.from_pretrained(path, dtype=torch.float32 if train else torch.float16, attn_implementation="sdpa")
    return tok, m.cuda().train(train)


class _Batches(Dataset):
    def __init__(self, texts, batches, tok):
        self.texts, self.batches, self.tok = texts, batches, tok

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        enc = self.tok([self.texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return idx, enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def encode(texts, tok, model, bs=1024, workers=16) -> np.ndarray:
    """L2-normalised fp16 embeddings. Length-sorted batches, tokenised in DataLoader workers."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    dl = DataLoader(_Batches(texts, batches, tok), batch_size=None, num_workers=workers, prefetch_factor=4)
    out = np.empty((len(texts), model.config.hidden_size), np.float16)
    model.eval()
    for idx, ids, am in dl:
        ids, am = ids.cuda(non_blocking=True), am.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            e = mean_pool(model(input_ids=ids, attention_mask=am).last_hidden_state, am)
        out[idx] = F.normalize(e.float(), dim=-1).half().cpu().numpy()
    return out
