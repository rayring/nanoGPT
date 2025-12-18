# 1. 输出设置
out_dir = "grok/out"
eval_interval = 200  # 每n步评估一次
eval_iters = 40  # 评估时跑多少个batch
log_interval = 40  # 每n步打印一次日志
always_save_checkpoint = False  # 不用一直存模型，省空间

# 2. 数据设置
dataset = "grok"
gradient_accumulation_steps = 1
batch_size = 256  # 字符级序列更长，适当减小 batch
block_size = 20

# 3. 模型设置 (非常小的 Transformer)
n_layer = 4
n_head = 6
n_embd = 384
dropout = 0.0
bias = False  # 去掉 bias 更有利于数学任务

# 4. 优化器设置 (Grokking 的秘诀)
learning_rate = 1e-3
max_iters = 10000  # 训练步数要足够长！
lr_decay_iters = max_iters
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1.2  # 极强的正则化，迫使参数"简化"

# 5. 设备
device = "cuda"
compile = False  # 小模型编译反而慢
