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
token_pad = "~"


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
    # 1. 加载数据
    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode="r")

    # 2. 对齐切片
    n_samples = (len(data) - 1) // block_size
    ix = torch.randint(0, n_samples, (batch_size,)) * block_size

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

    # 4. 动态 Mask 逻辑
    # 目标：只训练 "=" 之后，且包含 "E" 的部分，忽略 Padding 和问题部分。

    # cumsum(is_eq) > 0 : 表示当前位置已经在 "=" 之后 (或者是 "=" 本身)
    # cumsum(is_end) == 0: 表示当前位置还没有遇到 "E" (或者是 "E" 之前)

    # 逻辑推演：
    # InputX :  ... =  4  6  E  P ...
    # cum_eq :  ... 1  1  1  1  1 ... (从 = 开始变1)
    # cum_end:  ... 0  0  0  1  1 ... (从 E 开始变1)
    # mask   :  ... T  T  T  F  F ...
    # TargetY:  ... 4  6  E  P  P ...
    # ValidY :      ^  ^  ^ (保留预测 4, 6, E)

    is_eq = x == eq_id
    is_end = x == end_id
    mask = (torch.cumsum(is_eq, dim=1) > 0) & (torch.cumsum(is_end, dim=1) == 0)

    # 应用 Mask
    y[~mask] = ignore_id

    # 5. 最后一位保护
    y[:, -1] = ignore_id

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
