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
    # 1. 加载数据 (保持原有的 memmap 逻辑以防内存泄漏)
    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode="r")

    # 2. 按 block 获取
    max_ix = (len(data) - block_size) // block_size
    ix = torch.randint(0, max_ix, (batch_size,)) * block_size

    # 转换 Tensor
    to_tensor = lambda data_slice: torch.from_numpy(data_slice.astype(np.int64))
    x_stack = [to_tensor(data[i : i + block_size]) for i in ix]
    y_stack = [to_tensor(data[i + 1 : i + 1 + block_size]) for i in ix]

    x = torch.stack(x_stack)
    y = torch.stack(y_stack)

    # 3. 获取关键 Token ID
    stoi = get_meta()["stoi"]
    eq_id = stoi[token_eq]
    end_id = stoi[token_end]
    ignore_id = stoi[token_ignore]

    # 4. 应用 Mask
    y[:, : (6 - 1)] = ignore_id
    y[:, -1:] = ignore_id

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

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
