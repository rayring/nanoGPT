import os
import pickle
import random
import numpy as np

# 假设这些 token 定义在 grok.utils 里，这里直接使用
# 如果没有，请手动定义: token_end='E', token_eq='=', token_ignore='_', token_pad='P'
from grok.utils import get_batch, token_end, token_ignore, token_eq, token_pad
from grok.config import block_size

os.makedirs("data/grok", exist_ok=True)
assert block_size >= 20  # 确认足以容纳最长的 99*99*99=970299
random.seed(42)  # 保证可复现


def one_sample(nums: list, ops: list) -> str:
    """
    nums: 数字列表，例如 [1, 2, 3]
    ops: 符号列表，例如 ['+', '*']
    返回: "1+2*3=+7EPPPP..."
    """
    # 1. 构建算式字符串
    expression = str(nums[0])
    for i, op in enumerate(ops):
        expression += f"{op}{str(nums[i+1])}"

    # 2. 计算结果
    c = int(eval(expression))

    # 3. 格式化答案
    sign = "+" if c >= 0 else "-"
    answer = f"{sign}{abs(c)}"
    answerr_reversed = answer[::-1]
    text = f"{expression}{token_eq}{answerr_reversed}{token_end}"

    # 5. Padding填充，实现定长
    if len(text) > block_size:
        text = text[:block_size]

    return text.ljust(block_size, token_pad)


MAX_NUM = 100
OPS = ["+", "-", "*"]
samples = []

# 1. 遍历所有 1-op 情况 (保证基础算术能力覆盖)
for a in range(MAX_NUM):
    for b in range(MAX_NUM):
        for op in OPS:
            samples.append(one_sample([a, b], [op]))

# 2. 随机采样 2-op 情况 (混合运算)
# 数量: 追加 30,000 条，保持数据集平衡
# 组合: a op1 b op2 c
for _ in range(30000):
    nums = [random.randint(0, MAX_NUM - 1) for _ in range(3)]
    ops = [random.choice(OPS) for _ in range(2)]
    samples.append(one_sample(nums, ops))

# 打乱数据
random.shuffle(samples)

data_str = "".join(samples) + token_pad
chars = sorted(list(set(data_str)))
VOCAB_SIZE = len(chars)
assert VOCAB_SIZE < 64 * 1024

stoi = {**{ch: i for i, ch in enumerate(chars)}, token_ignore: -1}
itos = {**{i: ch for i, ch in enumerate(chars)}, -1: token_ignore}
encode = lambda s: [stoi[c] for c in s]

# 划分训练集和验证集
n = int(0.9 * len(samples))  # 90% 训练，10% 验证
train_text = "".join(samples[:n])
val_text = "".join(samples[n:])

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)
train_ids.tofile("data/grok/train.bin")
val_ids.tofile("data/grok/val.bin")


meta = {
    "vocab_size": VOCAB_SIZE,
    "itos": itos,
    "stoi": stoi,
}
with open("data/grok/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"VocabSize: {VOCAB_SIZE}")
print(f"Vocabs: {chars}")
print(f"TrainTokenCount: {len(train_ids)}")
print(f"ValTokenCount: {len(val_ids)}")
print("-" * 40)

# 数据预览
print(f"SampleCount: {len(samples)}")
for i in range(5):
    print(f"{samples[i]}")
print("-" * 40)

# Batch 预览
x, y = get_batch("train", block_size, 5, "cpu")
for i in range(x.shape[0]):
    # print(f"> {i}")
    print(f"X:", "".join(map(itos.get, x[i].tolist())) + "_")
    print(f"Y:", "_" + "".join(map(itos.get, y[i].tolist())))
    # print(f"X:", x[i].tolist())
    # print(f"Y:", y[i].tolist())
