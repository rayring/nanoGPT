import os
import random
import numpy as np
from utils import get_batch

OPS = ["+", "-", "*"]

# 创建目录
os.makedirs("data/grok", exist_ok=True)


def one_sample(a: int, op: str, b: int) -> str:
    assert a in range(0, 100)
    assert b in range(0, 100)

    if op == "+":
        c = a + b
    elif op == "-":
        c = a - b
    elif op == "*":
        c = a * b
    else:
        raise ValueError(f"unknown op: {op}")

    # 每个数字/符号都是 1 个 token：例如 123 -> '1','2','3'
    sign = "+" if c >= 0 else "-"
    return f"{a:02d}{op}{b:02d}={sign}{abs(c):04d}E"


samples = []
for a in range(100):
    for b in range(100):
        for op in OPS:
            samples.append(one_sample(a, op, b))

data_str = "".join(samples)
chars = sorted(list(set(data_str)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)


def encode(s: str):
    return [stoi[c] for c in s]


# 数据预览
preview_indices = random.sample(range(len(samples)), 10)
for idx in preview_indices:
    sample = samples[idx].rstrip("E")
    print(f"{idx:>8}: {sample}")
print("-" * 40)

# 打乱数据
random.seed(42)
random.shuffle(samples)

# 划分训练集和验证集 (50% / 50%)
# 这种极端划分是为了强迫模型去"猜"剩下的规律，而不是死记硬背
n = int(0.5 * len(samples))
train_text = "".join(samples[:n])
val_text = "".join(samples[n:])

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)

# 保存为 nanoGPT 能读取的 .bin 格式
train_ids.tofile("data/grok/train.bin")
val_ids.tofile("data/grok/val.bin")

# 保存 meta 信息 (用于编解码，虽然这里只是简单的ID映射)
import pickle

meta = {
    "vocab_size": VOCAB_SIZE,
    "itos": itos,
    "stoi": stoi,
}
with open("data/grok/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

assert VOCAB_SIZE < 64 * 1024

print(
    f"VOCAB_SIZE={VOCAB_SIZE}, TrainTokenCount: {len(train_ids)}, ValTokenCount: {len(val_ids)}"
)


print("-" * 40)
x, y = get_batch("train", 12, 5, "cpu")
for i in range(x.shape[0]):
    print(f"> {i}")
    print(f"X:", "".join(map(itos.get, x[i].tolist())))
    print(f"Y:", "".join(map(itos.get, y[i].tolist())))
