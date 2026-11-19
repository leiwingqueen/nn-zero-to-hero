"""
makemore part4: 手写反向传播 —— 练习版
=======================================

对应课程：lectures/makemore/makemore_part4_backprop.ipynb

part3 学会了 BatchNorm 和初始化。本节不引入新模型，而是拆开 PyTorch 的
autograd：沿着前向计算图逐个变量推导梯度，再把交叉熵和 BatchNorm 的反向
传播化简，最后用自己写的梯度训练字符级 MLP。

本节练习：
  1. 对前向过程中每个中间变量逐个手写梯度，并与 PyTorch autograd 比较；
  2. 推导 softmax + cross entropy 的简洁梯度；
  3. 推导 BatchNorm 对输入的梯度；
  4. 将手写梯度接入完整训练循环。

使用方法：
  1. names.txt 放在本文件同目录（没有的话见 load_words 里的下载命令）。
  2. 补全所有 TODO 标记的表达式/代码。
  3. 运行 `python3 makemore_part4_backprop_exercise.py`。

梯度比较使用固定随机种子和小批次；完成 Exercise 1 后，各项应与 autograd
近似一致。完整训练默认 200000 步，CPU 运行需要一些时间，可先调小 MAX_STEPS。
"""

import os
import random

import torch
import torch.nn.functional as F

SEED = 2147483647
BLOCK_SIZE = 3
VOCAB_SIZE = 27
N_EMB = 10
N_HIDDEN = 64
BN_EPS = 1e-5
MAX_STEPS = 200000


# ---------------------------------------------------------------------------
# 0. 数据与字符表（前几节已练过，这里直接给出）
# ---------------------------------------------------------------------------

def load_words(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先下载数据集：\n"
            "  curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
        )
    with open(path, "r") as f:
        return f.read().splitlines()


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    return stoi, {i: s for s, i in stoi.items()}


def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def prepare_data(path=None):
    words = load_words(path)
    stoi, itos = build_vocab(words)
    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi)
    Xte, Yte = build_dataset(words[n2:], stoi)
    return (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte), stoi, itos


# ---------------------------------------------------------------------------
# 1. 逐节点手写反向传播
# ---------------------------------------------------------------------------

def cmp(name, dt, tensor):
    """把手写梯度 dt 与 tensor.grad 比较，打印最大绝对误差。"""
    exact = torch.equal(dt, tensor.grad)
    close = torch.allclose(dt, tensor.grad)
    maxdiff = (dt - tensor.grad).abs().max().item()
    print(f"{name:15s} | exact: {str(exact):5s} | approximate: {str(close):5s} | maxdiff: {maxdiff:.3g}")


def make_gradient_demo(Xtr, Ytr, vocab_size=VOCAB_SIZE, seed=SEED):
    """搭建课程中的小型 BN-MLP 前向图，返回所有中间量供梯度练习使用。"""
    n_emb, n_hidden = N_EMB, N_HIDDEN
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_emb), generator=g)
    W1 = torch.randn((n_emb * BLOCK_SIZE, n_hidden), generator=g) * (5 / 3) / (n_emb * BLOCK_SIZE) ** 0.5
    b1 = torch.randn(n_hidden, generator=g) * 0.1
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, n_hidden), generator=g) * 0.1
    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad_(True)

    g_batch = torch.Generator().manual_seed(seed + 1)
    batch_size = 32
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g_batch)
    Xb, Yb = Xtr[ix], Ytr[ix]
    n = batch_size

    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmeani = hprebn.mean(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)  # unbiased sample variance
    bnvar_inv = (bnvar + BN_EPS) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    # 展开 cross entropy，练习者可以从这个表达式开始推导。
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    for p in parameters:
        p.grad = None
    intermediates = [logprobs, probs, counts, counts_sum, counts_sum_inv,
                     norm_logits, logit_maxes, logits, h, hpreact, bnraw,
                     bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
                     embcat, emb]
    for t in intermediates:
        t.retain_grad()
    loss.backward()

    cache = locals().copy()
    cache["parameters"] = parameters
    cache["intermediates"] = intermediates
    return cache


def exercise_manual_backward(c):
    """Exercise 1：逐个中间节点反向传播，完成 TODO 后对照 autograd。"""
    # 交叉熵索引与均值
    # print(f'logprobs.shape:{c["logprobs"].shape}')
    # 这里可以简单理解，L=(a+b+c)/n 的公式下，可以替换为 L=a/n+b/n+c/n，这里对a,b,c求导都是1/n
    # 然后因为dlogprobs里面只有命中Yb的下标的中间变量才会影响L的结果，所以要去 dlogprobs[range(c["n"]), c["Yb"]]
    dlogprobs = torch.zeros_like(c["logprobs"])
    dlogprobs[range(c["n"]), c["Yb"]] = -1.0 / c["n"]

    # TODO：按计算图反向经过 log、乘法、sum、exp、减最大值等操作。
    dprobs = dlogprobs / c["probs"]
    # print(
    #    f'probs.shape:f{c["probs"].shape},counts.shape:{c["counts"].shape},counts_sum_inv.shape:{c["counts_sum_inv"].shape}')
    # 这里可以理解 counts_sum_inv 其实是做了扩展的，size[32,1]直接扩展成[32,27]
    dcounts_sum_inv = (dprobs * c["counts"]).sum(1, keepdim=True)
    # dcounts=dprobs * torch.ones_like(c["counts"]) * c["counts_sum_inv"]
    # 因为dprobs和c["counts"]的shape是相同的，因此 torch.ones_like(c["counts"])这个可以省略掉
    dcounts = dprobs * c["counts_sum_inv"]
    # counts_sum_inv = counts_sum ** -1
    dcounts_sum = dcounts_sum_inv * (-c["counts_sum"] ** -2)
    # counts_sum = counts.sum(1, keepdim=True)
    # counts_sum的shape是 [32,1],counts的shape是[32,27]
    # 可以简单理解c=a+b+c，c对a/b/c求导都是1，我们唯一要做的是需要生成一个[32,27]全部为1的tensor
    dcounts += dcounts_sum * torch.ones_like(c["counts"])
    # counts = norm_logits.exp()
    dnorm_logits = dcounts * c["counts"]
    # norm_logits = logits - logit_maxes
    # 注意，这里必须用clone，不然会被后面的计算影响
    dlogits = dnorm_logits.clone()
    # print(f'logits:{c['logits'].shape},logit_maxes:{c['logit_maxes'].shape}')
    dlogit_maxes = -dnorm_logits.sum(1, keepdim=True)
    # logit_maxes = logits.max(1, keepdim=True).values
    dlogits += dlogit_maxes * F.one_hot(c["logits"].max(1).indices, num_classes=c["logits"].shape[1])
    # print(f'logits:{c['logits'].shape},h:{c['h'].shape},W2:{c['W2'].shape},b2:{c['b2'].shape}')
    # logits = h @ W2 + b2，这里需要手写一遍加深记忆
    dh = dlogits @ c["W2"].T
    dW2 = c["h"].T @ dlogits
    db2 = dlogits.sum(0)
    # h = torch.tanh(hpreact)
    dhpreact = dh * (1 - c['h'] ** 2)
    # hpreact = bngain * bnraw + bnbias
    # print(
    #    f'hpreact:{c['hpreact'].shape},bngain:{c['bngain'].shape},bnraw:{c['bnraw'].shape},bnbias:{c['bnbias'].shape}')
    dbngain = (dhpreact * c["bnraw"]).sum(0, keepdim=True)
    dbnraw = dhpreact * c["bngain"]
    dbnbias = dhpreact.sum(0, keepdim=True)
    # bnraw = bndiff * bnvar_inv
    # print(f'bnraw:{c['bnraw'].shape},bndiff:{c['bndiff'].shape},bnvar_inv:{c['bnvar_inv'].shape}')
    dbndiff = dbnraw * c["bnvar_inv"]
    dbnvar_inv = (dbnraw * c["bndiff"]).sum(0, keepdim=True)
    # bnvar_inv = (bnvar + BN_EPS) ** -0.5
    dbnvar = dbnvar_inv * (-0.5 * (c["bnvar"] + BN_EPS) ** -1.5)
    # bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)  # unbiased sample variance
    # print(f'bnvar:{c["bnvar"].shape},bndiff2:{c["bndiff2"].shape}')
    dbndiff2 = dbnvar * torch.ones_like(c["bndiff2"]) / (c["n"] - 1)
    # bndiff2 = bndiff ** 2
    dbndiff = dbndiff2 * (2 * c["bndiff"])
    dhprebn = ...
    dbnmeani = ...
    dhprebn = ...
    dembcat = ...
    dW1 = ...
    db1 = ...
    demb = ...

    dC = torch.zeros_like(c["C"])
    # TODO：embedding 是查表操作；把重复字符位置的梯度累加回 C。
    # 提示：遍历 Xb 的 batch 维和 block_size 维，累加 demb[k, j]。
    ...

    manual = {
        "logprobs": dlogprobs, "probs": dprobs,
        "counts_sum_inv": dcounts_sum_inv, "counts_sum": dcounts_sum,
        "counts": dcounts, "norm_logits": dnorm_logits,
        "logit_maxes": dlogit_maxes, "logits": dlogits, "h": dh,
        "W2": dW2, "b2": db2, "hpreact": dhpreact,
        "bngain": dbngain, "bnbias": dbnbias, "bnraw": dbnraw,
        "bnvar_inv": dbnvar_inv, "bnvar": dbnvar,
        "bndiff2": dbndiff2, "bndiff": dbndiff,
        "bnmeani": dbnmeani, "hprebn": dhprebn,
        "embcat": dembcat, "W1": dW1, "b1": db1,
        "emb": demb, "C": dC,
    }
    tensors = {"C": c["C"], "W1": c["W1"], "b1": c["b1"],
               "W2": c["W2"], "b2": c["b2"], "bngain": c["bngain"],
               "bnbias": c["bnbias"]}
    tensors.update({name: value for name, value in zip(
        ["logprobs", "probs", "counts", "counts_sum", "counts_sum_inv",
         "norm_logits", "logit_maxes", "logits", "h", "hpreact", "bnraw",
         "bnvar_inv", "bnvar", "bndiff2", "bndiff", "hprebn", "bnmeani",
         "embcat", "emb"], c["intermediates"])})
    for name, grad in manual.items():
        cmp(name, grad, tensors[name])
    return manual


# ---------------------------------------------------------------------------
# 2. 合并 cross entropy 的反向传播
# ---------------------------------------------------------------------------

def exercise_cross_entropy_backward(logits, targets):
    """Exercise 2：返回 mean cross entropy 对 logits 的梯度，shape 与 logits 相同。"""
    n = targets.shape[0]
    dlogits = F.softmax(logits, dim=1)
    # TODO：对每行正确类别减去 1，再除以 batch 大小。
    ...
    return dlogits


# ---------------------------------------------------------------------------
# 3. 合并 BatchNorm 的反向传播
# ---------------------------------------------------------------------------

def exercise_batchnorm_backward(dout, x, gamma, eps=BN_EPS):
    """Exercise 3：BatchNorm 使用 unbiased=True 的方差，返回对 x 的梯度。"""
    n = x.shape[0]
    mean = x.mean(0, keepdim=True)
    var = x.var(0, keepdim=True, unbiased=True)
    var_inv = (var + eps) ** -0.5
    xhat = (x - mean) * var_inv
    # TODO：将逐步求导结果化简成对输入的一条表达式。
    # 课程推导对应：gamma * var_inv / n * (...)
    dx = ...
    return dx


# ---------------------------------------------------------------------------
# 4. 端到端手写梯度训练
# ---------------------------------------------------------------------------

def init_model(vocab_size=VOCAB_SIZE, n_emb=N_EMB, n_hidden=200, seed=SEED):
    """初始化课程中的 MLP + BatchNorm 参数。"""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_emb), generator=g)
    W1 = torch.randn((n_emb * BLOCK_SIZE, n_hidden), generator=g) * (5 / 3) / (n_emb * BLOCK_SIZE) ** 0.5
    b1 = torch.randn(n_hidden, generator=g) * 0.1
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, n_hidden), generator=g) * 0.1
    params = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in params:
        p.requires_grad_(True)
    return params


def train_manual(Xtr, Ytr, params, steps=MAX_STEPS, batch_size=32, seed=SEED, verbose=True):
    """用手写梯度训练，返回每步 loss 的 log10。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    g = torch.Generator().manual_seed(seed)
    lossi = []
    with torch.no_grad():
        for i in range(steps):
            ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
            Xb = Xtr[ix]
            Yb = Ytr[ix]
            n = batch_size

            emb = C[Xb]
            embcat = emb.view(emb.shape[0], -1)
            hprebn = embcat @ W1 + b1
            bnmean = hprebn.mean(0, keepdim=True)
            bnvar = hprebn.var(0, keepdim=True, unbiased=True)
            bnvar_inv = (bnvar + BN_EPS) ** -0.5
            bnraw = (hprebn - bnmean) * bnvar_inv
            hpreact = bngain * bnraw + bnbias
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Yb)

            # 手写反向传播
            dlogits = exercise_cross_entropy_backward(logits, Yb)
            dh = dlogits @ W2.T
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(0)
            dhpreact = (1 - h ** 2) * dh
            dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
            dbnbias = dhpreact.sum(0, keepdim=True)
            dhprebn = exercise_batchnorm_backward(dhpreact, hprebn, bngain)
            dembcat = dhprebn @ W1.T
            dW1 = embcat.T @ dhprebn
            db1 = dhprebn.sum(0)
            demb = dembcat.view(emb.shape)
            dC = torch.zeros_like(C)
            # TODO：将 embedding 梯度累加回 C（重复索引必须累加）。
            ...

            grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
            lr = 0.1 if i < steps // 2 else 0.01
            # TODO：按梯度下降更新每个参数，注意当前处于 no_grad 环境。
            ...

            if verbose and i % 10000 == 0:
                print(f"{i:7d}/{steps:7d}: {loss.item():.4f}")
            lossi.append(loss.log10().item())
    return lossi


@torch.no_grad()
def split_loss(X, Y, params, batch_stats=None):
    """使用给定数据集统计量评估 loss；batch_stats 为训练集的 (mean, var)。"""
    C, W1, b1, W2, b2, bngain, bnbias = params
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    if batch_stats is None:
        mean = hprebn.mean(0, keepdim=True)
        var = hprebn.var(0, keepdim=True, unbiased=True)
    else:
        mean, var = batch_stats
    hpreact = bngain * (hprebn - mean) * (var + BN_EPS) ** -0.5 + bnbias
    logits = torch.tanh(hpreact) @ W2 + b2
    return F.cross_entropy(logits, Y).item()


def main():
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte), stoi, itos = prepare_data()
    print(f"数据集：train={len(Xtr)}, dev={len(Xdev)}, test={len(Xte)}, vocab={len(stoi)}")

    # 小批次梯度核对（补全 Exercise 1 后取消注释）
    demo = make_gradient_demo(Xtr, Ytr, vocab_size=len(stoi))
    exercise_manual_backward(demo)

    # 两个简化公式与 autograd 的核对
    logits = torch.randn(8, len(stoi), requires_grad=True)
    targets = torch.randint(0, len(stoi), (8,))
    F.cross_entropy(logits, targets).backward()
    dlogits = exercise_cross_entropy_backward(logits.detach(), targets)
    assert torch.allclose(dlogits, logits.grad, atol=1e-7), "cross entropy 梯度不匹配"

    x = torch.randn(8, 5, requires_grad=True)
    gamma = torch.randn(1, 5)
    y = gamma * (x - x.mean(0, keepdim=True)) / torch.sqrt(x.var(0, keepdim=True, unbiased=True) + BN_EPS)
    upstream = torch.randn_like(y)
    y.backward(upstream)
    dx = exercise_batchnorm_backward(upstream, x.detach(), gamma)
    assert torch.allclose(dx, x.grad, atol=1e-6), "BatchNorm 梯度不匹配"
    print("cross entropy / BatchNorm 梯度公式校验通过")

    params = init_model(vocab_size=len(stoi))
    # lossi = train_manual(Xtr, Ytr, params, steps=MAX_STEPS)
    # 训练结束后，用 train split 的整体均值和方差校准 BatchNorm，再评估 dev/test。
    # 请参考 split_loss 与课程 notebook 完成校准和采样。


if __name__ == "__main__":
    main()
