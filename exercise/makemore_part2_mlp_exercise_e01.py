"""
makemore part2: MLP 字符级语言模型 —— 练习版
=============================================

对应课程：lectures/makemore/makemore_part2_mlp.ipynb
论文原型：Bengio et al. 2003, "A Neural Probabilistic Language Model"

part1 的 bigram 模型只能看前 1 个字符，想看更多上下文，表格会指数爆炸
（看 3 个字符就是 27^3 = 19683 行，绝大多数格子的计数都是 0）。
本节课的解法：把每个字符先映射成一个低维稠密向量（embedding），
再把上下文的向量拼接起来送进一个 MLP（多层感知机）来预测下一个字符。

网络结构（block_size=3, n_emb=10, n_hidden=200）：

    输入 3 个字符下标  (N, 3)
        │  C[X]                       查表：每个字符 -> 10 维向量
        ▼
    embedding          (N, 3, 10)
        │  .view(N, 30)               把 3 个向量拼成一条 30 维
        ▼
    (N, 30) @ W1 + b1  -> tanh        隐藏层，200 个神经元
        ▼
    (N, 200) @ W2 + b2 -> logits      输出层，27 个类别
        ▼
    cross_entropy(logits, Y)          loss

本节课的新知识点（也是本练习要练的）：
  - embedding 查表：C[X] 的高级索引用法
  - view / 内存布局：为什么 .view 比 torch.cat 好
  - F.cross_entropy：为什么它比手写 softmax + log 更好（数值稳定 + 更快）
  - minibatch：用一小撮样本的近似梯度换取更多的迭代次数
  - 学习率搜索：用指数扫描找到合适的 lr，以及后期 learning rate decay
  - train / dev / test 三分法：如何判断欠拟合还是过拟合

使用方法：
  1. 数据集 names.txt 已在本目录（没有的话见 load_words 里的下载命令）。
  2. 把所有标记为 `TODO` 的函数补全（现在是 `raise NotImplementedError`）。
  3. 运行 `python makemore_part2_mlp_exercise.py`，脚本会逐节做断言校验。

耗时参考：CPU 上训练 10 万步约 25 秒，整个脚本 1 分钟内跑完，不需要 GPU。
"""

import os
import random

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# 固定随机种子，保证结果可复现（课程里用的就是这个数）
SEED = 2147483646

# 词表大小：26 个字母 + 1 个特殊 token '.'（同时表示开始和结束）
VOCAB_SIZE = 27

# 上下文长度：用前 block_size 个字符预测下一个字符
BLOCK_SIZE = 5

# 每个字符 embedding 的维度（课程里先用 2 方便可视化，后来调成 10）
N_EMB = 20

# 隐藏层神经元个数
N_HIDDEN = 300

# 训练部署
TRAIN_STEP = 400_000

LEARNING_RATE = 0.1

BATCH_SIZ = 64


# ---------------------------------------------------------------------------
# 0. 读数据 + 字符表（part1 已经练过，这里直接给出实现）
# ---------------------------------------------------------------------------

def load_words(path=None):
    """读入 names.txt，返回名字列表。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先下载数据集：\n"
            "  curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
        )
    return open(path, "r").read().splitlines()


def build_vocab(words):
    """字符 <-> 整数 的双向映射，'.' 固定为 0。"""
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


# ---------------------------------------------------------------------------
# 1. 构造数据集：滑动窗口
# ---------------------------------------------------------------------------

def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    """
    把单词列表摊平成 (X, Y)：
      X: shape (N, block_size) 的 LongTensor，每行是 block_size 个字符下标（上下文）
      Y: shape (N,)            的 LongTensor，对应要预测的下一个字符下标

    要求：
      - 每个单词从 context = [0] * block_size 开始（全是 '.'，表示"还没有任何字符"）；
      - 遍历 w + '.'（末尾补结束符）里的每个字符 ch：
          先把 (context, ix) 存进数据集，再滑动窗口 context = context[1:] + [ix]。

    以 'emma'、block_size=3 为例，应该产生 5 条样本：
        ... ---> e        X=[0, 0, 0]     Y=5
        ..e ---> m        X=[0, 0, 5]     Y=13
        .em ---> m        X=[0, 5, 13]    Y=13
        emm ---> a        X=[5, 13, 13]   Y=1
        mma ---> .        X=[13, 13, 1]   Y=0

    注意最后一条：上下文 'mma' 要预测结束符 '.'，这是模型学会"何时停下"的关键。
    """
    # 实现数据集构造
    X = []
    Y = []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:]
            context.append(ix)
    return torch.tensor(X), torch.tensor(Y)


def split_words(words, seed=42):
    """
    把单词列表按 80% / 10% / 10% 切成 train / dev / test 三份（这里直接给出实现）。

    为什么要三份（课程里的重点）：
      - train  训练集：用来算梯度、更新参数；
      - dev    验证集（又叫 validation）：用来调超参数（层数、维度、学习率、正则强度…）；
      - test   测试集：只在最后看一次，用来报告最终性能。
        如果你反复在 test 上调参，test 就"被污染"了，它的分数会虚高。

      train loss ≈ dev loss  -> 欠拟合（underfitting），模型容量还不够，可以加大网络；
      train loss << dev loss -> 过拟合（overfitting），模型开始背答案了。
    """
    words = list(words)
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


# ---------------------------------------------------------------------------
# 2. 参数初始化
# ---------------------------------------------------------------------------

def init_params(seed=SEED, block_size=BLOCK_SIZE, n_emb=N_EMB, n_hidden=N_HIDDEN):
    """
    初始化并返回参数列表 [C, W1, b1, W2, b2]，且全部 requires_grad_()。

    形状：
      C  : (VOCAB_SIZE, n_emb)          embedding 查表矩阵，第 i 行 = 字符 i 的向量
      W1 : (block_size * n_emb, n_hidden)   隐藏层权重，输入是拼接后的上下文向量
      b1 : (n_hidden,)
      W2 : (n_hidden, VOCAB_SIZE)       输出层权重
      b2 : (VOCAB_SIZE,)

    要求：
      - 全部用 torch.randn(..., generator=g) 初始化，g = torch.Generator().manual_seed(seed)，
        且按 C, W1, b1, W2, b2 的顺序依次抽，这样结果才可复现；
      - 返回前把每个参数设成 requires_grad = True。

    默认配置下总参数量应为 11897 个（是不是比 27^3 的表格小多了？）。
    """
    # 实现参数初始化
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(VOCAB_SIZE, n_emb, generator=g)
    W1 = torch.randn(block_size * n_emb, n_hidden, generator=g) * (5 / 3) / (block_size * n_emb) ** 0.5
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn(n_hidden, VOCAB_SIZE, generator=g) * 0.01
    b2 = torch.randn(VOCAB_SIZE, generator=g) * 0
    parameters = [C, W1, b1, W2, b2]
    for param in parameters:
        param.requires_grad = True
    return parameters


# ---------------------------------------------------------------------------
# 3. 前向传播
# ---------------------------------------------------------------------------

def forward(X, params):
    """
    前向传播，输入 X 是 (N, block_size) 的下标张量，返回 logits，shape (N, VOCAB_SIZE)。

    三步：
      1) emb = C[X]
         这是 PyTorch 的高级索引：用一个 (N, block_size) 的整数张量去索引 (27, n_emb)
         的矩阵，结果是 (N, block_size, n_emb)。等价于对每个下标取 C 的对应行，
         也等价于 one-hot @ C —— 但查表比矩阵乘法快得多。

      2) h = torch.tanh(emb.view(N, -1) @ W1 + b1)
         .view(N, -1) 把 (N, block_size, n_emb) 摊平成 (N, block_size * n_emb)。
         ⚠️ 为什么用 view 而不是 torch.cat([emb[:,0,:], emb[:,1,:], ...], 1)？
            因为 view 只是换一个"怎么读这块内存"的说明书（stride/shape），
            不复制任何数据，O(1)；而 cat 会新分配内存并拷贝。
         ⚠️ 为什么用 X.shape[0] 或 -1 而不是写死 32？
            因为训练用 minibatch（32 行），评估用整个数据集（18 万行），要能自适应。

      3) logits = h @ W2 + b2
         注意这里返回的是 logits（未归一化的分数），不是概率。
    """

    """
    网络结构（block_size=3, n_emb=10, n_hidden=200）：

    输入 3 个字符下标  (N, 3)
        │  C[X]                       查表：每个字符 -> 10 维向量
        ▼
    embedding          (N, 3, 10)
        │  .view(N, 30)               把 3 个向量拼成一条 30 维
        ▼
    (N, 30) @ W1 + b1  -> tanh        隐藏层，200 个神经元
        ▼
    (N, 200) @ W2 + b2 -> logits      输出层，27 个类别
        ▼
    cross_entropy(logits, Y)          loss
    """
    # 实现前向传播
    N = X.shape[0]
    C = params[0]
    W1 = params[1]
    b1 = params[2]
    W2 = params[3]
    b2 = params[4]
    # (N,block_size,n_emb)
    emb = C[X]
    # (N,n_hidden)
    hidden = torch.tanh(emb.view(N, -1) @ W1 + b1)
    logits = hidden @ W2 + b2
    return logits


# ---------------------------------------------------------------------------
# 4. 训练：minibatch + 学习率衰减
# ---------------------------------------------------------------------------

def train(Xtr, Ytr, params, steps=100000, batch_size=32, lr=0.1, lr_decay_at=0.6,
          decay_factor=0.1, seed=SEED, verbose=True):
    """
    minibatch 梯度下降，原地更新 params，返回每步 loss 组成的 list（用于画曲线）。

    每一步做四件事：
      1) 采 minibatch：
           ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
         为什么要 minibatch？一次算 18 万条样本的梯度要 ~1 秒，一次算 32 条只要 ~0.1 毫秒。
         minibatch 的梯度方向没那么准，但"走 1000 步歪一点的路"远胜过"走 1 步准的路"。

      2) forward：logits = forward(Xtr[ix], params)
                  loss   = F.cross_entropy(logits, Ytr[ix])
         ⚠️ 为什么用 F.cross_entropy 而不是手写 counts/counts.sum() 再取 log？
            a) 它不会为中间结果（counts、probs）分配张量，前向更快、反向的表达式也更简单；
            b) 数值稳定：内部会先减掉每行的最大值再取 exp。
               手写版本遇到较大的 logit（比如 100）时 exp 会溢出成 inf，整个 loss 变 nan。

      3) backward：先把所有 p.grad 置为 None，再 loss.backward()
         （PyTorch 的梯度是累加的，不清零会把上一步的梯度叠进来）

      4) update：p.data += -lr * p.grad
         lr 在训练进行到 lr_decay_at 比例处衰减为 lr * decay_factor
         （learning rate decay：先大步快速下降，后期小步精调）。

    默认配置下，跑完 10 万步后 minibatch loss 会在 2.1~2.2 上下跳动。
    ⚠️ minibatch loss 抖动很大是正常的（每次只看 32 条样本），
       判断模型好坏要用下面 split_loss 在完整 train/dev 上算的 loss。
    """
    # 实现训练循环
    g = torch.Generator().manual_seed(seed)
    N = Xtr.shape[0]
    losses = []
    decay = lr_decay_at * steps
    for i in range(steps):
        # ix是(batch_size,)
        ix = torch.randint(0, N, (batch_size,), generator=g)
        # Xtr[ix]的维度是 (batch_size,block_size)
        logit = forward(Xtr[ix], params)
        loss = F.cross_entropy(logit, Ytr[ix])
        # backward
        for p in params:
            p.grad = None
        loss.backward()
        losses.append(loss.item())
        if i == 0:
            print(f"first step loss:{loss.item():.4f}")
        # update
        lr_i = lr if i < decay else lr * decay_factor
        for p in params:
            p.data += -lr_i * p.grad
    return losses


@torch.no_grad()
def split_loss(X, Y, params):
    """
    在完整的某个 split 上计算 loss，返回 float。

    就是 forward 一次再 F.cross_entropy，注意：
      - 函数上面的 @torch.no_grad() 装饰器告诉 PyTorch 不用记录计算图，
        省显存也更快（评估阶段不需要反向传播）；
      - 返回 .item() 而不是张量。
    """
    # 实现 split 上的 loss 评估
    logit = forward(X, params)
    loss = F.cross_entropy(logit, Y)
    return loss.item()


# ---------------------------------------------------------------------------
# 5. 学习率搜索（课程里"怎么知道 lr 该设多少"的那一节）
# ---------------------------------------------------------------------------

def find_lr(Xtr, Ytr, steps=1000, lr_min=-3, lr_max=0, seed=SEED):
    """
    在 [10^lr_min, 10^lr_max] 区间内指数扫描学习率，返回 (lrs, losses) 两个 list。

    做法：
      - lre = torch.linspace(lr_min, lr_max, steps); lrs = 10 ** lre
        ⚠️ 为什么在指数上均匀取值？因为学习率的影响是乘性的：
           0.001 -> 0.01 的差别，和 0.1 -> 1.0 的差别是同一个量级的事。
      - 用 init_params() 重新初始化一套全新的参数（不要污染正式训练的参数）；
      - 第 i 步用 lrs[i] 作为学习率跑一次 minibatch 更新，记录 loss。

    然后把 (lrs, losses) 画出来：一开始 loss 缓慢下降，某处开始剧烈震荡甚至上升，
    "震荡开始前的那个 lr" 就是比较合适的取值。这就是课程里得到 lr≈0.1 的方法。
    """
    # 实现学习率扫描
    batch_size = 32
    g = torch.Generator().manual_seed(seed)
    lre = torch.linspace(lr_min, lr_max, steps)
    lrs = (10 ** lre).tolist()
    losses = []
    params = init_params(seed=seed)
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        logit = forward(Xtr[ix], params)
        loss = F.cross_entropy(logit, Ytr[ix])
        # backward
        for p in params:
            p.grad = None
        loss.backward()
        losses.append(loss.log10().item())
        lr = lrs[i]
        # update
        for p in params:
            p.data += -lr * p.grad
    return lrs, losses


# ---------------------------------------------------------------------------
# 6. 采样
# ---------------------------------------------------------------------------

def sample(params, itos, num=20, block_size=BLOCK_SIZE, seed=SEED + 10):
    """
    从训练好的模型里采样 num 个名字，返回字符串列表（不含结尾的 '.'）。

    过程和 part1 一样，只是"给定上下文求概率分布"换成了走一遍 MLP：
      - context = [0] * block_size
      - 循环：
          logits = forward(torch.tensor([context]), params)   # 注意要包成 batch，(1, block_size)
          probs  = F.softmax(logits, dim=1)
          ix     = torch.multinomial(probs, num_samples=1, generator=g).item()
          context = context[1:] + [ix]                        # 滑动窗口
          遇到 ix == 0（'.'）就结束，否则把字符追加到结果里。

    这个模型采样出来的名字（carmah、amelle、khi …）明显比 part1 的 bigram 更像名字了。
    """
    # 实现采样
    g = torch.Generator().manual_seed(seed)
    name_list = []
    for i in range(num):
        context = [0] * block_size
        name = []
        while True:
            X = torch.tensor([context])
            logits = forward(X, params)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:]
            context.append(ix)
            if ix == 0:
                name_list.append(''.join(name))
                break
            name.append(itos[ix])
    return name_list


# ---------------------------------------------------------------------------
# 主流程：逐节自检
# ---------------------------------------------------------------------------

def main():
    words = load_words()
    stoi, itos = build_vocab(words)
    print(f"数据集大小: {len(words)}，示例: {words[:5]}")

    # --- 1. 数据集构造 ---
    tr_words, dev_words, te_words = split_words(words)
    Xtr, Ytr = build_dataset(tr_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)
    Xte, Yte = build_dataset(te_words, stoi)

    # --- 2. 参数初始化 ---
    params = init_params()
    nparams = sum(p.nelement() for p in params)
    print(f"      参数量: {nparams}")

    # --- 4. 学习率搜索 ---
    lrs, losses = find_lr(Xtr, Ytr, steps=1000)
    plt.plot(lrs, losses)
    plt.show()

    # --- 5. 训练 ---
    print(f"      开始训练 {TRAIN_STEP} 步（CPU 约 25 秒）...")
    lossi = train(Xtr, Ytr, params, steps=TRAIN_STEP, batch_size=BLOCK_SIZE, lr=LEARNING_RATE)
    l_tr = split_loss(Xtr, Ytr, params)
    l_dev = split_loss(Xdev, Ydev, params)
    print(f"      train loss = {l_tr:.4f}   dev loss = {l_dev:.4f}, l_dev - l_tr = {l_dev - l_tr:.4f}")
    # part1 的 bigram 只能做到 2.45 左右，MLP 明显更好
    assert l_tr < 2.30, f"train loss 期望低于 2.30，实际 {l_tr:.4f}"
    assert l_dev < 2.32, f"dev loss 期望低于 2.32，实际 {l_dev:.4f}"
    # train 和 dev 几乎一样 -> 现在还是欠拟合，加大网络还能继续提升
    assert l_dev - l_tr < 0.15, "train/dev 差距过大，可能过拟合了"
    print("[5/6] 训练 OK")

    # --- 6. 采样 ---
    names = sample(params, itos, num=10)
    print("      采样结果:", names)
    assert len(names) == 10, "应返回 10 个名字"
    assert all(n and all(ch in "abcdefghijklmnopqrstuvwxyz" for ch in n) for n in names), \
        "采样结果应是非空的纯字母字符串（不要把结束符 '.' 带出来）"
    print("[6/6] 采样 OK")

    print(f"\n全部通过 🎉  dev loss {l_dev:.4f}（part1 的 bigram 是 2.4544）")
    print("test loss 留到最后一次性揭晓:", f"{split_loss(Xte, Yte, params):.4f}")
    print("\n接着往下做进阶练习（见文件末尾 E01 / E02 / E03）👇")


# ---------------------------------------------------------------------------
# 进阶练习（Karpathy 在视频简介里留的作业）
# ---------------------------------------------------------------------------
#
# E01: 调超参数，把 dev loss 打到 2.17 以下，越低越好。可以动的旋钮：
#        - N_EMB（embedding 维度，10 -> 更大）
#        - N_HIDDEN（隐藏层宽度，200 -> 300/500）
#        - BLOCK_SIZE（上下文长度，3 -> 4/5/8）
#        - batch_size、总步数、学习率与衰减策略
#      每改一次只在 dev 上看结果，test 留到最后。
#      记录下每组超参对应的 dev loss，你会发现"加大网络"的收益是递减的。
#
# E02: 目前初始 loss 高达 20+，而理论上随机猜的 loss 应该是 -log(1/27) ≈ 3.30。
#      原因是 W2/b2 初始化太大，导致 logits 分布很宽、softmax 极度自信却猜错。
#      试着把 W2 乘以 0.01、b2 乘以 0（甚至直接置零），再看看：
#        - 初始 loss 是不是降到了 3.3 附近？
#        - 训练曲线开头那段"曲棍球柄"形状是不是消失了？
#        - 同样步数下最终 loss 是不是更低？
#      （这正是 part3 "Activations & Gradients" 整节课要展开讲的东西。）
#
# E03: 读一遍 Bengio et al. 2003 的论文，看看有哪些本课没实现的想法值得一试。
#      https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
#      提示：论文里有一条从输入直连输出的 skip connection。
#
# E04（可选，可视化）: 当 N_EMB=2 时，把 C 的 27 个点画在平面上并标注字母：
#        plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
#        for i in range(C.shape[0]):
#            plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center")
#      你会看到元音 a/e/i/o/u 聚成一簇，q、x 这种稀有字符被甩到角落 ——
#      网络自己学出了"字符的相似性"，这就是 embedding 能泛化到没见过的上下文的原因。

if __name__ == "__main__":
    main()
