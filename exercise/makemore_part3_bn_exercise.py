"""
makemore part3: 激活值、梯度与 BatchNorm —— 练习版
=====================================================

对应课程：lectures/makemore/makemore_part3_bn.ipynb
论文原型：
  - Kaiming He et al. 2015, "Delving Deep into Rectifiers"（Kaiming 初始化）
  - Ioffe & Szegedy 2015, "Batch Normalization"

part2 的 MLP 能训练，但它"能训练"其实有点侥幸：初始 loss 高达 20+，
tanh 隐藏层大量饱和，网络前几千步都在白白浪费。网络只有 1 层时问题还不致命，
一旦叠到 5 层、10 层，激活值和梯度会逐层爆炸或消失，网络就根本训不动了。

本节课不引入新的模型能力，而是学会"看懂网络内部的健康状况"，并用三种手段修好它：

    问题                              诊断方法                         修复
    ─────────────────────────────────────────────────────────────────────────────
    初始 loss 远大于 -log(1/27)        对比 expected_init_loss          W2 * 0.01, b2 = 0
    tanh 饱和（|h| 接近 1，梯度 ≈ 0）  tanh_saturation / dead_neurons   缩小 W1
    深层网络激活值逐层缩小/爆炸        activation_stats 逐层看 std      Kaiming init（gain / sqrt(fan_in)）
    对初始化依然很敏感                 —                                BatchNorm
    学习率是否合适                     update:data ratio ≈ 1e-3         调 lr

本节课的新知识点（也是本练习要练的）：
  - 初始 loss 应该是多少，以及 softmax "自信地犯错" 的问题
  - tanh 饱和与"死神经元"：为什么 |h|≈1 时梯度会消失
  - Kaiming 初始化：std = gain / sqrt(fan_in)，以及 tanh 的 gain 为什么是 5/3
  - BatchNorm：训练时用 batch 统计量、推理时用 running 统计量；bias 为什么变得多余
  - 把代码 "PyTorch 化"：自己写 Linear / BatchNorm1d / Tanh 三个模块
  - 网络体检：激活值分布、梯度分布、grad:data 比例、update:data 比例

使用方法：
  1. 数据集 names.txt 已在本目录（没有的话见 load_words 里的下载命令）。
  2. 把所有标记为 `TODO` 的函数/方法补全（现在是 `raise NotImplementedError`）。
  3. 运行 `python makemore_part3_bn_exercise.py`，脚本会逐节做断言校验。

耗时参考：CPU 上整个脚本约 1 分钟，不需要 GPU。
"""

import math
import os
import random

import torch
import torch.nn.functional as F

# 固定随机种子，保证结果可复现（课程里用的就是这个数）
SEED = 2147483647

VOCAB_SIZE = 27
BLOCK_SIZE = 3
N_EMB = 10
N_HIDDEN = 200


# ---------------------------------------------------------------------------
# 0. 读数据 + 构造数据集（part1 / part2 已经练过，这里直接给出实现）
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


def build_dataset(words, stoi, block_size=BLOCK_SIZE):
    """滑动窗口构造 (X, Y)，同 part2。"""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def split_words(words, seed=42):
    """80% / 10% / 10% 切成 train / dev / test，同 part2。"""
    words = list(words)
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


# ---------------------------------------------------------------------------
# 1. 初始 loss：softmax "自信地犯错"
# ---------------------------------------------------------------------------

def expected_init_loss(vocab_size=VOCAB_SIZE):
    """
    返回一个"刚初始化、什么都没学到"的网络，理论上应该有的 loss（float）。

    思路：网络还没学过任何东西，最合理的状态是对 27 个字符一视同仁，
    即输出均匀分布 p = 1/27。此时对任何一个正确答案，交叉熵都是 -log(1/27)。

    27 个字符时结果约为 3.2958。
    如果你的网络初始 loss 是 27（part2 就是这样），说明它在一开始就
    "非常自信地给出了错误答案" —— logits 的数值太大，softmax 被推向了极端。
    """
    # TODO: 实现 expected_init_loss
    raise NotImplementedError("expected_init_loss")


def init_mlp(scheme="kaiming", seed=SEED, n_emb=N_EMB, n_hidden=N_HIDDEN, block_size=BLOCK_SIZE):
    """
    初始化 part2 的单隐藏层 MLP，返回参数列表 [C, W1, b1, W2, b2]（全部 requires_grad）。

    依次用 torch.randn(..., generator=g) 抽 C, W1, b1, W2, b2（顺序固定，保证可复现），
    g = torch.Generator().manual_seed(seed)。然后根据 scheme 对它们做缩放：

      "naive"       不做任何缩放（即 part2 的做法）。
      "fix_softmax" 让输出层一开始"不那么自信"：W2 *= 0.01，b2 *= 0。
                    ⚠️ 为什么 W2 不直接置 0？全 0 的话输出确实是均匀分布，
                       但所有隐藏神经元收到的梯度完全一样，对称性无法打破。
                       用很小的随机数就够了。
      "fix_tanh"    在 fix_softmax 基础上，再让隐藏层别饱和：W1 *= 0.2，b1 *= 0.01。
                    0.2 是课程里"肉眼调出来的魔法数字"。
      "kaiming"     在 fix_softmax 基础上，用有原则的方式缩放 W1：
                      W1 *= kaiming_std(fan_in=n_emb*block_size, nonlinearity="tanh")
                    b1 *= 0.01。（kaiming_std 在第 2 节实现）

    提示：缩放要在设置 requires_grad 之前做，或者放在 torch.no_grad() 里做，
          否则缩放后的张量不再是叶子节点，.grad 会是 None。
    """
    # TODO: 实现 init_mlp
    raise NotImplementedError("init_mlp")


def mlp_forward(X, params):
    """
    part2 的前向传播（直接给出），额外把隐藏层也返回，方便做诊断。
    返回 (logits, hpreact, h)：
      hpreact  tanh 之前的值（pre-activation），shape (N, n_hidden)
      h        tanh 之后的值，shape (N, n_hidden)
    """
    C, W1, b1, W2, b2 = params
    emb = C[X]
    hpreact = emb.view(emb.shape[0], -1) @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return logits, hpreact, h


# ---------------------------------------------------------------------------
# 2. tanh 饱和 + Kaiming 初始化
# ---------------------------------------------------------------------------

def tanh_saturation(h, thresh=0.97):
    """
    返回 h 中 |h| > thresh 的元素占比（float，0~1）。

    为什么关心这个？tanh 的反向传播是  grad_in = (1 - h**2) * grad_out。
    当 |h| 接近 1 时，(1 - h**2) ≈ 0，梯度在这里被"掐断"了 ——
    无论 loss 想让它怎么变，这个神经元前面的 W1、b1、C 都收不到信号。

    课程里用 plt.hist(h.view(-1).tolist(), 50) 和
    plt.imshow(h.abs() > 0.99, cmap='gray') 来直观感受，
    naive 初始化下直方图几乎全堆在 -1 和 +1 两端。
    """
    # TODO: 实现 tanh_saturation
    raise NotImplementedError("tanh_saturation")


def dead_neurons(h, thresh=0.99):
    """
    返回"死神经元"的个数（int）：对 batch 里的 **每一个** 样本都饱和（|h| > thresh）的神经元。

    h 的 shape 是 (N, n_hidden)，第 j 列是第 j 个神经元在 N 个样本上的输出。
    如果某一列全部饱和，那么这个神经元对任何输入都传不回梯度，它永远学不到东西了 ——
    这就是"死神经元"（dead neuron）。ReLU 在输入恒为负时也会出现同样的问题。

    提示：(...).all(dim=0) 可以沿 batch 维做"全部为真"的判断。
    """
    # TODO: 实现 dead_neurons
    raise NotImplementedError("dead_neurons")


def kaiming_std(fan_in, nonlinearity="tanh"):
    """
    返回 Kaiming 初始化的标准差：gain / sqrt(fan_in)（float）。

    推导直觉：y = x @ W，x 的每个分量方差为 1，W 的每个元素方差为 s^2，
    那么 y 的每个分量是 fan_in 项之和，方差 = fan_in * s^2。
    想让 y 的方差仍然是 1，就要 s = 1 / sqrt(fan_in)。

    但非线性会"压缩"分布（tanh 把大的值压扁，ReLU 砍掉一半），
    每层都会让方差缩小一点，层数一多就消失了。于是乘一个 gain 补回来：
        linear   gain = 1
        tanh     gain = 5/3
        relu     gain = sqrt(2)      （ReLU 丢掉一半，方差减半，所以乘 sqrt(2)）
    这些数字和 torch.nn.init.calculate_gain 一致。
    未知的 nonlinearity 请 raise ValueError。
    """
    # TODO: 实现 kaiming_std
    raise NotImplementedError("kaiming_std")


# ---------------------------------------------------------------------------
# 3. 手写 BatchNorm（单隐藏层 MLP 版）
# ---------------------------------------------------------------------------

BN_EPS = 1e-5
BN_MOMENTUM = 0.001  # 课程手写版用的是 0.001；batch 只有 32，统计量很抖，所以动量取小


def init_bn_mlp(seed=SEED, n_emb=N_EMB, n_hidden=N_HIDDEN, block_size=BLOCK_SIZE):
    """
    初始化带 BatchNorm 的 MLP（直接给出），返回 (params, buffers)。

      params  = [C, W1, W2, b2, bngain, bnbias]    需要梯度、会被训练
      buffers = {"mean": ..., "var": ...}          不需要梯度、用滑动平均更新

    注意 W1 后面 **没有 b1** 了：BatchNorm 会先减掉 batch 均值，
    b1 加上去马上就被减掉，它的梯度恒为 0，完全是摆设。
    它的角色由 BatchNorm 自己的 bnbias 接替。
    """
    g = torch.Generator().manual_seed(seed)
    fan_in = n_emb * block_size
    C = torch.randn((VOCAB_SIZE, n_emb), generator=g)
    W1 = torch.randn((fan_in, n_hidden), generator=g) * kaiming_std(fan_in, "tanh")
    W2 = torch.randn((n_hidden, VOCAB_SIZE), generator=g) * 0.01
    b2 = torch.zeros(VOCAB_SIZE)
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    params = [C, W1, W2, b2, bngain, bnbias]
    for p in params:
        p.requires_grad = True
    buffers = {"mean": torch.zeros((1, n_hidden)), "var": torch.ones((1, n_hidden))}
    return params, buffers


def bn_mlp_forward(X, params, buffers, training=True, momentum=BN_MOMENTUM, eps=BN_EPS):
    """
    带 BatchNorm 的前向传播，返回 logits，shape (N, VOCAB_SIZE)。

        emb     = C[X]  -> view 成 (N, block_size * n_emb)
        hpreact = embcat @ W1                     # 注意：没有 + b1
        ── BatchNorm ─────────────────────────────────────────────
        training=True:
            mean = hpreact.mean(0, keepdim=True)  # (1, n_hidden)，每个神经元在 batch 上的均值
            var  = hpreact.var(0, keepdim=True)   # (1, n_hidden)
            并且在 torch.no_grad() 下 **原地** 更新 buffers 里的滑动平均：
              buffers["mean"] = (1 - momentum) * buffers["mean"] + momentum * mean
              buffers["var"]  = (1 - momentum) * buffers["var"]  + momentum * var
        training=False:
            mean, var 直接取 buffers["mean"], buffers["var"]，不做任何更新
        hpreact = bngain * (hpreact - mean) / sqrt(var + eps) + bnbias
        ──────────────────────────────────────────────────────────
        h      = tanh(hpreact)
        logits = h @ W2 + b2

    ⚠️ 为什么训练和推理要分两套统计量？
       训练时：mean/var 来自当前 batch，于是一个样本的输出会受同 batch 其他样本影响。
              这是 BatchNorm 的"副作用"，但它像数据增强一样带来了一点正则化效果。
       推理时：往往一次只喂 1 个样本，根本没法算 batch 统计量，
              所以要用训练过程中累积下来的"全体训练集统计量的估计"。

    ⚠️ 为什么 buffer 的更新要放在 no_grad 里？它不是参数，不该进计算图。
    ⚠️ 为什么要加 eps？如果某个神经元在 batch 内恰好输出全一样，var=0，会除以 0。
    """
    # TODO: 实现 bn_mlp_forward
    raise NotImplementedError("bn_mlp_forward")


@torch.no_grad()
def calibrate_bn(Xtr, params):
    """
    训练结束后，把整个训练集过一遍，**精确** 计算 W1 输出的均值和方差，
    返回 {"mean": (1, n_hidden), "var": (1, n_hidden)}。

    这是课程里先讲的"笨办法"：训练完再单独做一步校准。
    滑动平均（buffers）就是为了省掉这一步 —— 训练过程中顺手估出来。
    主流程会验证两者非常接近。
    """
    # TODO: 实现 calibrate_bn
    raise NotImplementedError("calibrate_bn")


def train_loop(Xtr, Ytr, parameters, forward_fn, steps, batch_size=32, lr=0.1,
               lr_decay_at=0.5, seed=SEED, print_every=10000):
    """
    通用 minibatch 训练循环（part2 已经练过，直接给出）。
    forward_fn(Xb) -> logits，返回每步 loss 的 list。
    """
    g = torch.Generator().manual_seed(seed)
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(forward_fn(Xtr[ix]), Ytr[ix])
        for p in parameters:
            p.grad = None
        loss.backward()
        cur_lr = lr if i < steps * lr_decay_at else lr * 0.1
        for p in parameters:
            p.data += -cur_lr * p.grad
        if print_every and i % print_every == 0:
            print(f"      {i:7d}/{steps:7d}: {loss.item():.4f}")
        lossi.append(loss.item())
    return lossi


# ---------------------------------------------------------------------------
# 4. PyTorch 化：自己写 Linear / BatchNorm1d / Tanh
# ---------------------------------------------------------------------------
#
# 下面三个类的接口和 torch.nn 里同名模块保持一致：
#   - __call__(x) 做前向，并把输出存到 self.out（方便后面做诊断，正式代码不会这么干）
#   - parameters() 返回需要训练的张量列表
#
# 有了它们，"搭一个 N 层网络"就变成了"写一个列表"，和 nn.Sequential 一个思路。

class Linear:

    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        """
        weight: (fan_in, fan_out)，用 torch.randn(..., generator=generator) / fan_in**0.5 初始化
                （即 gain=1 的 Kaiming；需要别的 gain 时由外部再乘）
        bias:   (fan_out,)，初始化为 0；bias=False 时为 None
        """
        # TODO: 实现 Linear.__init__
        raise NotImplementedError("Linear.__init__")

    def __call__(self, x):
        """out = x @ weight (+ bias)，存进 self.out 并返回。"""
        # TODO: 实现 Linear.__call__
        raise NotImplementedError("Linear.__call__")

    def parameters(self):
        # TODO: 实现 Linear.parameters
        raise NotImplementedError("Linear.parameters")


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """
        需要的属性：
          eps, momentum
          training = True               训练/推理模式开关
          gamma = ones(dim)             可训练的缩放
          beta  = zeros(dim)            可训练的平移
          running_mean = zeros(dim)     buffer，不训练
          running_var  = ones(dim)      buffer，不训练

        momentum=0.1 是 PyTorch 的默认值。手写版用 0.001 是因为 batch 太小统计量太抖；
        batch 大一些时 0.1 就够了。
        """
        # TODO: 实现 BatchNorm1d.__init__
        raise NotImplementedError("BatchNorm1d.__init__")

    def __call__(self, x):
        """
        和第 3 节 bn_mlp_forward 里的 BatchNorm 部分完全一样，只是：
          - 统计量来自 self.training 决定的那一套；
          - 训练模式下用 self.momentum 在 no_grad 里更新 running_mean / running_var；
          - 输出 self.out = gamma * xhat + beta，存起来并返回。
        x 的 shape 是 (N, dim)，均值和方差沿第 0 维（batch 维）计算。
        """
        # TODO: 实现 BatchNorm1d.__call__
        raise NotImplementedError("BatchNorm1d.__call__")

    def parameters(self):
        # TODO: 实现 BatchNorm1d.parameters
        raise NotImplementedError("BatchNorm1d.parameters")


class Tanh:

    def __call__(self, x):
        # TODO: 实现 Tanh.__call__
        raise NotImplementedError("Tanh.__call__")

    def parameters(self):
        return []


def build_deep_net(n_layers=5, n_hidden=100, use_bn=True, gain=5 / 3, seed=SEED,
                   n_emb=N_EMB, block_size=BLOCK_SIZE):
    """
    搭一个 n_layers 个隐藏层的深层 MLP，返回 (C, layers, parameters)。

    结构（use_bn=True 时）：
        Linear(n_emb*block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden,         n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        ...（共 n_layers 组）
        Linear(n_hidden, VOCAB_SIZE, bias=False), BatchNorm1d(VOCAB_SIZE)
    use_bn=False 时去掉所有 BatchNorm1d，Linear 带 bias（bias=True）。

    要求：
      1) g = torch.Generator().manual_seed(seed)；先抽 C = randn((VOCAB_SIZE, n_emb))，
         再按顺序构造各层（Linear 用 generator=g）。
      2) 在 torch.no_grad() 下调整初始化：
           - 除最后一层外，所有 Linear 的 weight *= gain
           - 最后一层"不要那么自信"：use_bn 时最后的 BatchNorm1d 的 gamma *= 0.1，
             否则最后一个 Linear 的 weight *= 0.1
      3) parameters = [C] + 所有层的参数，全部 requires_grad = True。

    ⚠️ Linear 后面紧跟 BatchNorm 时为什么 bias=False？见 init_bn_mlp 的说明。
    """
    # TODO: 实现 build_deep_net
    raise NotImplementedError("build_deep_net")


def deep_forward(X, C, layers):
    """深层网络前向传播（直接给出）：embedding -> 拼接 -> 逐层调用。"""
    emb = C[X]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return x


def set_training(layers, mode):
    """切换所有层的训练/推理模式（直接给出），相当于 nn.Module 的 .train() / .eval()。"""
    for layer in layers:
        if hasattr(layer, "training"):
            layer.training = mode


# ---------------------------------------------------------------------------
# 5. 网络体检：激活值、梯度、更新幅度
# ---------------------------------------------------------------------------

def activation_stats(layers, thresh=0.97):
    """
    对每个 Tanh 层，统计它上一次前向的输出 layer.out，返回 list，每个元素是 dict：
        {"mean": float, "std": float, "saturated": float}
    saturated 是 |out| > thresh 的占比（复用 tanh_saturation）。

    健康的网络：各层 std 大致相同（不逐层缩小，也不逐层放大），饱和率在 5% 左右。
    课程里还会用 torch.histogram 把每层分布画成曲线叠在一起看。
    """
    # TODO: 实现 activation_stats
    raise NotImplementedError("activation_stats")


def grad_stats(layers):
    """
    对每个 Tanh 层，统计 layer.out.grad 的标准差，返回 float 的 list。

    要能拿到中间结果的梯度，前向之后、backward 之前需要调用
        layer.out.retain_grad()
    （PyTorch 默认只给叶子节点保留 .grad）。这一步由主流程的 diagnose() 负责。

    健康的网络：各层梯度的 std 也大致相同。
    如果从后往前逐层变小 -> 梯度消失；逐层变大 -> 梯度爆炸。
    """
    # TODO: 实现 grad_stats
    raise NotImplementedError("grad_stats")


def grad_data_ratio(p):
    """
    返回 p.grad.std() / p.data.std()（float）。

    它衡量"梯度相对于参数本身有多大"。如果某个参数的这个比例比别人大很多，
    同样的学习率下它会被更新得过猛。课程里发现最后一层的这个比例特别大，
    这是因为我们刻意把最后一层初始化得很小（*0.1）。
    """
    # TODO: 实现 grad_data_ratio
    raise NotImplementedError("grad_data_ratio")


def update_data_ratio(p, lr):
    """
    返回 log10( (lr * p.grad).std() / p.data.std() )（float）。

    这是课程最后讲的、判断学习率是否合适的经验指标：
    每一步参数的"变化量"相对于参数本身的大小。经验值是 **约 1e-3，即 log10 ≈ -3**：
      - 远小于 -3：学习率太小，参数几乎不动，训练太慢；
      - 远大于 -3：学习率太大，参数每步都被剧烈改写，训练不稳定。
    课程里把每一步每个参数的这个值都记下来画成曲线，看它们是否收敛在 -3 附近。
    """
    # TODO: 实现 update_data_ratio
    raise NotImplementedError("update_data_ratio")


def diagnose(X, Y, C, layers, parameters):
    """跑一次 forward + backward 并返回 (activation_stats, grad_stats)（直接给出）。"""
    logits = deep_forward(X, C, layers)
    loss = F.cross_entropy(logits, Y)
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()
    return activation_stats(layers), grad_stats(layers)


# ---------------------------------------------------------------------------
# 6. 采样
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(C, layers, itos, num=20, block_size=BLOCK_SIZE, seed=SEED + 10):
    """
    从深层网络采样 num 个名字，返回字符串列表（不含结尾的 '.'）。

    和 part2 一样，只是前向换成 deep_forward(torch.tensor([context]), C, layers)。

    ⚠️ 这里一次只喂 1 个样本 —— 如果 BatchNorm 还处在训练模式，
       对 1 个样本算方差会得到 nan（unbiased var 要除以 N-1 = 0）。
       所以采样前必须 set_training(layers, False)。这一步由调用方负责。
    """
    # TODO: 实现 sample
    raise NotImplementedError("sample")


@torch.no_grad()
def eval_loss(X, Y, forward_fn):
    return F.cross_entropy(forward_fn(X), Y).item()


# ---------------------------------------------------------------------------
# 主流程：逐节自检
# ---------------------------------------------------------------------------

def main():
    words = load_words()
    stoi, itos = build_vocab(words)
    tr_words, dev_words, te_words = split_words(words)
    Xtr, Ytr = build_dataset(tr_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)
    Xte, Yte = build_dataset(te_words, stoi)
    print(f"数据集: train {tuple(Xtr.shape)}  dev {tuple(Xdev.shape)}  test {tuple(Xte.shape)}")

    # --- 1. 初始 loss ---
    exp_loss = expected_init_loss()
    assert abs(exp_loss - 3.2958) < 1e-3, f"期望初始 loss ≈ 3.2958，实际 {exp_loss}"
    Xb, Yb = Xtr[:1000], Ytr[:1000]
    init_losses = {}
    for scheme in ["naive", "fix_softmax", "fix_tanh", "kaiming"]:
        params = init_mlp(scheme)
        assert len(params) == 5 and all(p.requires_grad and p.is_leaf for p in params), \
            f"[{scheme}] 应返回 5 个 requires_grad 的叶子张量"
        with torch.no_grad():
            logits, _, _ = mlp_forward(Xb, params)
        init_losses[scheme] = F.cross_entropy(logits, Yb).item()
        print(f"      {scheme:12s} 初始 loss = {init_losses[scheme]:.4f}")
    assert init_losses["naive"] > 15, "naive 初始化下 loss 应该非常大（part2 是 27 左右）"
    for scheme in ["fix_softmax", "fix_tanh", "kaiming"]:
        assert abs(init_losses[scheme] - exp_loss) < 0.05, \
            f"[{scheme}] 初始 loss 应接近 {exp_loss:.4f}，实际 {init_losses[scheme]:.4f}"
    p_naive, p_fix = init_mlp("naive"), init_mlp("fix_softmax")
    assert torch.equal(p_naive[0], p_fix[0]) and torch.equal(p_naive[1], p_fix[1]), \
        "C、W1 在 naive 和 fix_softmax 下应该一模一样（只缩放 W2、b2），检查随机数抽取顺序"
    assert p_fix[4].abs().sum().item() == 0, "fix_softmax 下 b2 应全为 0"
    print("[1/6] 初始 loss OK")

    # --- 2. tanh 饱和 + Kaiming ---
    t = torch.tensor([[0.5, -0.98, 0.999], [0.1, 0.995, -0.999]])
    assert abs(tanh_saturation(t) - 4 / 6) < 1e-6, "tanh_saturation 计算不对"
    assert dead_neurons(t) == 1, f"上面的例子里只有第 3 列是死神经元，你返回了 {dead_neurons(t)}"
    assert abs(kaiming_std(30, "tanh") - (5 / 3) / 30 ** 0.5) < 1e-9
    assert abs(kaiming_std(100, "relu") - math.sqrt(2) / 10) < 1e-9
    assert abs(kaiming_std(100, "linear") - 0.1) < 1e-9
    assert abs(kaiming_std(30, "tanh") - torch.nn.init.calculate_gain("tanh") / 30 ** 0.5) < 1e-9, \
        "你的 tanh gain 应该和 torch.nn.init.calculate_gain('tanh') 一致"
    try:
        kaiming_std(10, "sigmoid-ish")
        raise AssertionError("未知 nonlinearity 应该 raise ValueError")
    except ValueError:
        pass
    sat = {}
    for scheme in ["naive", "fix_tanh", "kaiming"]:
        with torch.no_grad():
            _, hpre, h = mlp_forward(Xtr[:32], init_mlp(scheme))
        sat[scheme] = (tanh_saturation(h), dead_neurons(h), hpre.std().item())
        print(f"      {scheme:9s} hpreact std {sat[scheme][2]:6.3f}  饱和率 {sat[scheme][0]:6.1%}  "
              f"死神经元 {sat[scheme][1]:3d}/{N_HIDDEN}")
    assert sat["naive"][0] > 0.5, "naive 初始化下隐藏层应该大面积饱和"
    # Kaiming 让 hpreact 的 std ≈ 5/3（故意略大于 1，抵消 tanh 的压缩），饱和率会比 fix_tanh 稍高
    assert sat["fix_tanh"][0] < 0.1 and sat["kaiming"][0] < 0.25, "修复后饱和率应明显下降"
    assert sat["kaiming"][1] == 0, "Kaiming 初始化下不应有死神经元"
    print("[2/6] tanh 饱和 + Kaiming OK")

    # --- 3. 手写 BatchNorm ---
    params, buffers = init_bn_mlp()
    logits = bn_mlp_forward(Xtr[:32], params, buffers, training=True)
    assert logits.shape == (32, VOCAB_SIZE)
    assert not buffers["mean"].requires_grad, "buffer 更新要放在 torch.no_grad() 里"
    assert buffers["mean"].abs().sum() > 0, "训练模式下应该更新 running mean"
    # BN 之后、tanh 之前的值每个神经元在 batch 上应该是 0 均值 1 方差（gain=1, bias=0 时）
    C, W1 = params[0], params[1]
    hpre = Xtr[:32]
    hpre = C[hpre].view(32, -1) @ W1
    bnout = (hpre - hpre.mean(0, keepdim=True)) / torch.sqrt(hpre.var(0, keepdim=True) + BN_EPS)
    expect = torch.tanh(bnout) @ params[2] + params[3]
    assert torch.allclose(logits, expect, atol=1e-5), "训练模式下的 BatchNorm 计算不对"
    snapshot = {k: v.clone() for k, v in buffers.items()}
    l1 = bn_mlp_forward(Xdev[:5], params, buffers, training=False)
    l2 = bn_mlp_forward(Xdev[:1], params, buffers, training=False)
    assert all(torch.equal(snapshot[k], buffers[k]) for k in buffers), "推理模式不应更新 buffers"
    assert torch.allclose(l1[:1], l2, atol=1e-6), "推理模式下，一个样本的输出不应受同 batch 其他样本影响"

    params, buffers = init_bn_mlp()
    steps = 50000
    print(f"      训练 BN-MLP {steps} 步（CPU 约 15 秒）...")
    train_loop(Xtr, Ytr, params, lambda xb: bn_mlp_forward(xb, params, buffers, True), steps)
    calib = calibrate_bn(Xtr, params)
    mean_err = (calib["mean"] - buffers["mean"]).abs().mean().item()
    std_rel = ((calib["var"].sqrt() - buffers["var"].sqrt()).abs() / calib["var"].sqrt()).mean().item()
    print(f"      running vs 校准: mean 平均差 {mean_err:.4f}，std 平均相对误差 {std_rel:.2%}")
    assert std_rel < 0.05, "running 统计量应和整个训练集上校准出的统计量非常接近"
    fwd_eval = lambda x: bn_mlp_forward(x, params, buffers, training=False)
    l_tr, l_dev = eval_loss(Xtr, Ytr, fwd_eval), eval_loss(Xdev, Ydev, fwd_eval)
    print(f"      train loss = {l_tr:.4f}   dev loss = {l_dev:.4f}")
    assert l_dev < 2.20, f"BN-MLP dev loss 期望低于 2.20，实际 {l_dev:.4f}"
    print("[3/6] 手写 BatchNorm OK")

    # --- 4. PyTorch 化 ---
    g = torch.Generator().manual_seed(SEED)
    lin = Linear(30, 100, bias=False, generator=g)
    assert lin.bias is None and len(lin.parameters()) == 1
    assert abs(lin.weight.std().item() - 30 ** -0.5) < 0.02, "Linear 的权重应初始化为 randn / sqrt(fan_in)"
    assert len(Linear(4, 3).parameters()) == 2 and Linear(4, 3).bias.abs().sum() == 0
    bn = BatchNorm1d(4)
    x = torch.randn(64, 4, generator=g) * 3 + 5
    y = bn(x)
    assert torch.allclose(y.mean(0), torch.zeros(4), atol=1e-5), "训练模式下 BN 输出每列均值应为 0"
    assert torch.allclose(y.std(0), torch.ones(4), atol=1e-3), "训练模式下 BN 输出每列标准差应为 1"
    assert torch.allclose(bn.running_mean.view(-1), 0.1 * x.mean(0), atol=1e-5), "running_mean 更新不对"
    bn.training = False
    y1 = bn(x[:1])
    assert torch.allclose(y1, (x[:1] - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps), atol=1e-5), \
        "推理模式应使用 running 统计量"
    t = Tanh()
    assert torch.equal(t(x), torch.tanh(x)) and t.out is not None

    C, layers, parameters = build_deep_net()
    nparams = sum(p.nelement() for p in parameters)
    print(f"      5 层 BN 网络参数量: {nparams}")
    assert len(layers) == 17, f"5 层 + BN 的网络应有 17 个 layer，实际 {len(layers)}"
    assert nparams == 47024, f"参数量期望 47024（课程同款），实际 {nparams}"
    assert all(p.requires_grad for p in parameters)
    assert torch.allclose(layers[-1].gamma, torch.full((VOCAB_SIZE,), 0.1)), "最后一层 BN 的 gamma 应乘 0.1"
    _, layers_nobn, params_nobn = build_deep_net(use_bn=False)
    assert len(layers_nobn) == 11 and sum(p.nelement() for p in params_nobn) == 46497, \
        "不带 BN 的网络应有 11 层、46497 个参数"
    print("[4/6] PyTorch 化 OK")

    # --- 5. 网络体检 ---
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=torch.Generator().manual_seed(SEED))
    Xb, Yb = Xtr[ix], Ytr[ix]
    print("      各配置下 5 个 Tanh 层的激活 std / 饱和率 / 梯度 std：")
    report = {}
    for name, kw in [("无BN gain=1  ", dict(use_bn=False, gain=1.0)),
                     ("无BN gain=5/3", dict(use_bn=False, gain=5 / 3)),
                     ("无BN gain=3  ", dict(use_bn=False, gain=3.0)),
                     ("有BN gain=5/3", dict(use_bn=True, gain=5 / 3))]:
        C_, layers_, params_ = build_deep_net(**kw)
        acts, grads = diagnose(Xb, Yb, C_, layers_, params_)
        assert len(acts) == 5 and len(grads) == 5, "应返回 5 个 Tanh 层的统计"
        report[name.strip()] = (acts, grads, params_)
        print(f"      {name}  std " + " ".join(f"{a['std']:.2f}" for a in acts)
              + "  | sat " + " ".join(f"{a['saturated']:4.0%}" for a in acts)
              + "  | grad " + " ".join(f"{gs:.1e}" for gs in grads))
    a1 = report["无BN gain=1"][0]
    assert a1[-1]["std"] < 0.6 * a1[0]["std"], "gain=1 时激活值应逐层缩小（tanh 不断压缩分布）"
    a3 = report["无BN gain=3"][0]
    assert a3[-1]["saturated"] > 0.4, "gain=3 时深层应大面积饱和"
    a53 = report["无BN gain=5/3"][0]
    assert all(0.5 < a["std"] < 0.8 for a in a53[1:]), "gain=5/3 时各层激活 std 应稳定在 0.65 附近"
    abn = report["有BN gain=5/3"][0]
    assert all(a["saturated"] < 0.1 for a in abn), "有 BN 时各层饱和率应保持很低"

    params_ = report["有BN gain=5/3"][2]
    W = [p for p in params_ if p.ndim == 2]
    r = grad_data_ratio(W[1])
    assert abs(r - (W[1].grad.std() / W[1].data.std()).item()) < 1e-9
    ud = update_data_ratio(W[1], 0.1)
    print(f"      第 1 个隐藏层权重 grad:data = {r:.2e}，lr=0.1 时 update:data = 10^{ud:.2f}")
    assert -4.5 < ud < -1.5, "lr=0.1 时 update:data 比例应在 1e-3 量级附近"
    assert update_data_ratio(W[1], 0.001) < ud - 1.9, "lr 缩小 100 倍，log10 比例应小 2"
    print("[5/6] 网络体检 OK")

    # --- 6. 训练深层网络 + 采样 ---
    C, layers, parameters = build_deep_net()
    steps = 30000
    print(f"      训练 5 层 BN 网络 {steps} 步（CPU 约 20 秒）...")
    set_training(layers, True)
    train_loop(Xtr, Ytr, parameters, lambda xb: deep_forward(xb, C, layers), steps)
    set_training(layers, False)
    fwd = lambda x: deep_forward(x, C, layers)
    l_tr, l_dev = eval_loss(Xtr, Ytr, fwd), eval_loss(Xdev, Ydev, fwd)
    print(f"      train loss = {l_tr:.4f}   dev loss = {l_dev:.4f}")
    assert l_dev < 2.25, f"深层网络 dev loss 期望低于 2.25，实际 {l_dev:.4f}"
    names = sample(C, layers, itos, num=10)
    print("      采样结果:", names)
    assert len(names) == 10
    assert all(n and all(ch in "abcdefghijklmnopqrstuvwxyz" for ch in n) for n in names), \
        "采样结果应是非空的纯字母字符串（采样前记得切到推理模式）"
    print("[6/6] 深层网络 + 采样 OK")

    print("\n全部通过 🎉")
    print("课程 loss 记录（单隐藏层，20 万步）：")
    print("  original 2.168 → fix softmax 2.13 → fix tanh 2.103 → kaiming 2.107 → batchnorm 2.105")
    print("\n接着往下做进阶练习（见文件末尾 E01 ~ E05）👇")


# ---------------------------------------------------------------------------
# 进阶练习（Karpathy 在视频简介里留的作业 + 课程中的实验）
# ---------------------------------------------------------------------------
#
# E01: 复现课程的 "loss log"。用 init_mlp 的四种 scheme 分别训练单隐藏层 MLP
#      （train_loop + mlp_forward，20 万步，lr_decay_at=0.5），记录 train/dev loss：
#          original 2.125/2.168 → fix softmax 2.07/2.13 → fix tanh 2.036/2.103
#          → kaiming 2.038/2.107 → batchnorm 2.067/2.105
#      体会：初始化修好以后，loss 曲线开头那段"曲棍球柄"消失了，省下的步数都用在了真正的学习上。
#
# E02: 把 build_deep_net 的 use_bn 设成 False，分别用 gain = 0.5 / 1 / 5/3 / 3 训练几千步，
#      对比 loss 下降速度。再把 Tanh 全部去掉（只剩 Linear 堆叠）：
#        - 此时 gain=1 才是对的（为什么？提示：没有非线性来"压缩"分布）；
#        - 整个网络退化成了一个线性模型，loss 能降到多少？
#
# E03: 在第 5 节的 diagnose 基础上，训练 1000 步，每一步把各 2 维参数的 update_data_ratio 存下来，
#      用 matplotlib 画成曲线，并画一条 y=-3 的参考线：
#          plt.plot([ud[j][i] for j in range(len(ud))]); plt.plot([0, len(ud)], [-3, -3], 'k')
#      然后把 lr 改成 0.001 和 1.0 各画一次，看看曲线会落到哪里。
#      再试试去掉最后一层的 "*0.1"，看最后一层的 grad:data 比例怎么变。
#
# E04（Karpathy 原题）: 把所有权重和偏置都初始化为 0，训练网络。
#      它还能学吗？如果能，学到的是什么？为什么只有一部分网络在学习？
#      提示：同一层所有神经元收到的梯度一模一样 —— "对称性"永远打不破。
#
# E05（Karpathy 原题）: BatchNorm 训练完之后，推理时其实可以把它"折叠"进前面的 Linear 层：
#          y = gamma * (x @ W - mean) / sqrt(var + eps) + beta
#            = x @ (W * gamma / sqrt(var + eps)) + (beta - gamma * mean / sqrt(var + eps))
#      写一个函数 fold_bn(linear, bn) 返回新的 Linear(bias=True)，用它替换每一对 (Linear, BatchNorm1d)，
#      验证折叠后的网络在 dev 上的输出和原网络完全一致（torch.allclose），且推理更快。
#      这正是工业界部署模型时的常规优化。
#
# E06（选做，可视化）: 打开 notebook 末尾的 BONUS 部分，跑一跑那两段"Linear / Linear+BN 的
#      前向和反向统计量"实验：把 w 不除以 sqrt(n) 时，x 的 std 会变成多少？为什么加了 BN
#      以后输出依然是 1，而且反向的梯度也被自动"缩放"回来了？


if __name__ == "__main__":
    main()

