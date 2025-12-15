# 1. 输出设置
out_dir = "grok/out"
eval_interval = 200  # 每n步评估一次
eval_iters = 200  # 评估时跑多少个batch
log_interval = 20  # 每n步打印一次日志
always_save_checkpoint = False  # 不用一直存模型，省空间

# 2. 数据设置
dataset = "grok"
gradient_accumulation_steps = 1
batch_size = 512  # 大一点的 batch size 有助于 grokking
block_size = 2  # 输入长度是 2 (a, b)，我们要预测第 3 个 (c)

# 3. 模型设置 (非常小的 Transformer)
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.0
bias = False  # 去掉 bias 更有利于数学任务

# 4. 优化器设置 (Grokking 的秘诀)
learning_rate = 1e-3
max_iters = 10000  # 训练步数要足够长！
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99
weight_decay = 1.2  # 极强的正则化，迫使参数"简化"

# 5. 设备
device = "cuda"
compile = False  # 小模型编译反而慢
