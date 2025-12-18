"""
Sample from a trained model
"""

import os
import sys
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
from grok.utils import get_meta, data_dir, token_eq, token_end
from grok.config import block_size

# -----------------------------------------------------------------------------
choice = int(sys.argv[1]) if len(sys.argv) > 1 else 0
start = ["FILE:prompts.txt", "BIN:train", "BIN:val"][choice]
out_dir = "grok/out"  # ignored if init_from is not 'resume'
num_samples = 3  # number of samples to draw
max_new_tokens = 6  # number of tokens generated in each sample
temperature = (
    0.01  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 233
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

# look for the meta pickle in case it is available in the dataset folder
meta = get_meta()
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


def get_prompts_by_file():
    with open(start[5:], "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    raw_lines = [p for p in raw_lines if p.strip() != ""]
    assert len(raw_lines) > 0

    prompts = []
    expected = []
    for line in raw_lines:
        assert token_eq in line
        left, right = line.split(token_eq, 1)
        prompts.append(left + token_eq)
        expected.append(right)

    return prompts, expected


def get_prompts_by_bin(source, n_grep):
    # sample full, aligned sequences from the dataset so we can split at '=' reliably
    bin_path = os.path.join(data_dir, f"{source}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    n_samples = len(data) // block_size
    ix = torch.randint(n_samples, (n_grep,))
    x = torch.stack(
        [
            torch.from_numpy(
                (data[i * block_size : (i + 1) * block_size]).astype(np.int64)
            )
            for i in ix.tolist()
        ]
    ).to(device)

    prompts = []
    expected = []
    for row in x:
        s = decode(row.tolist())
        left, right = s.split(token_eq, 1)
        assert right.endswith(token_end)
        prompts.append(left + token_eq)
        expected.append(right)

    return prompts, expected


if start.startswith("FILE:"):
    prompts, expected = get_prompts_by_file()
else:
    prompts, expected = get_prompts_by_bin(start[4:], 200)

prompt_ids = [encode(p) for p in prompts]
groups = {}
for idx, ids in enumerate(prompt_ids):
    groups.setdefault(len(ids), []).append(idx)

# run generation
with torch.no_grad():
    with ctx:
        results = {i: [] for i in range(len(prompts))}
        for k in range(num_samples):
            for seqlen, indices in sorted(groups.items()):
                prompt_groups = [prompt_ids[i] for i in indices]
                x = torch.tensor(prompt_groups, dtype=torch.long, device=device)
                y = model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                for row, i in enumerate(indices):
                    results[i].append(decode(y[row].tolist()))

        total = len(prompts) * num_samples
        correct = 0
        for i, prompt in enumerate(prompts):
            print(f"{i+1:04d} {prompt}{expected[i]}")
            print("-" * 20)
            for text in results[i]:
                completion = text[len(prompt) : len(prompt) + len(expected[i])]
                ok = completion == expected[i]
                correct += int(ok)
                print(
                    f"exp: {expected[i]} | got: {completion} | ok: {'🥑' if ok else '🍉'}"
                )
            print()

        acc = correct / total if total > 0 else 0.0
        print(f"accuracy: {correct}/{total} = {acc:.4f}")
