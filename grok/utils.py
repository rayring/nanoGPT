import os
import numpy as np
import pickle
import torch
from .config import dataset

data_dir = os.path.join("data", dataset)

_meta = None
_meta_path = os.path.join(data_dir, "meta.pkl")

token_end = "E"
token_eq = "="
token_ignore = "∙"


def get_meta():
    global _meta
    if _meta is None:
        if os.path.exists(_meta_path):
            with open(_meta_path, "rb") as f:
                _meta = pickle.load(f)
        else:
            raise ValueError(f"{_meta_path} not found")
    return _meta


def get_batch(split, block_size, batch_size, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # do standard next-token prediction
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    # 对不可预测的token，用token_ignore替换F
    stoi = get_meta()["stoi"]
    end_id = stoi[token_end]
    eq_id = stoi[token_eq]
    ignore_id = stoi[token_ignore]

    for i in range(batch_size):
        end_pos = (y[i] == end_id).nonzero(as_tuple=False)
        eq_pos = (y[i] == eq_id).nonzero(as_tuple=False)
        if end_pos.numel() == 0 or eq_pos.numel() == 0:
            continue
        end_pos = end_pos[0, 0].item()
        eq_pos = eq_pos[0, 0].item()
        if (end_pos > eq_pos and end_pos < block_size - 1) or (
            eq_pos == block_size - 1
        ):
            y[i, -1] = ignore_id

    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def estimate_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
):
    pred = logits.argmax(dim=-1)
    mask = targets.ne(ignore_index)
    if mask.any():
        correct = pred.eq(targets).logical_and(mask).sum().item()
        total = mask.sum().item()
        return correct, total

    return 0, 0
