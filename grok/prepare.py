import os
import torch
import random
import numpy as np

P = 37
VOCAB_SIZE = P

# 创建目录
os.makedirs("data/grok", exist_ok=True)

# 生成所有可能的输入组合 (a + b = c)
data = []
for a in range(P):
    for b in range(P):
        c = (a + b) % P
        data.append([a, b, c])

# 数据预览
preview_indices = random.sample(range(len(data)), 10)
for idx in preview_indices:
    sample = data[idx]
    print(f"{idx:>8}: {sample}")
print("-" * 40)

# 打乱数据
random.seed(42)
random.shuffle(data)
data = torch.tensor(data, dtype=torch.long)

# 划分训练集和验证集 (50% / 50%)
# 这种极端划分是为了强迫模型去"猜"剩下的规律，而不是死记硬背
n = int(0.5 * len(data))
train_data = data[:n]
val_data = data[n:]

# 保存为 nanoGPT 能读取的 .bin 格式
train_data.numpy().astype(np.uint16).tofile("data/grok/train.bin")
val_data.numpy().astype(np.uint16).tofile("data/grok/val.bin")

# 保存 meta 信息 (用于编解码，虽然这里只是简单的ID映射)
import pickle

meta = {
    "vocab_size": VOCAB_SIZE,
    "itos": {i: str(i) for i in range(VOCAB_SIZE)},
    "stoi": {str(i): i for i in range(VOCAB_SIZE)},
}
with open("data/grok/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(
    f"VOCAB_SIZE={VOCAB_SIZE}, 训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}"
)
