"""
makemore part1: bigram 语言模型 —— 练习版
=========================================

对应课程：lectures/makemore/makemore_part1_bigrams.ipynb

目标：用两种方式实现同一个 bigram（二元组）字符级语言模型，并验证它们等价。
  方式 A：统计计数法 —— 数出每个字符对出现的次数，归一化成概率表 P。
  方式 B：神经网络法 —— 一个 27x27 的权重矩阵 W，用梯度下降最小化负对数似然。
  最终两者的 loss 应该非常接近（约 2.45~2.47）。

使用方法：
  1. 准备数据集 names.txt（32033 个英文名字，每行一个），放在本文件同级目录，
     或从 https://raw.githubusercontent.com/karpathy/makemore/master/names.txt 下载。
  2. 把所有标记为 `TODO` 的地方补全（函数体里目前是 `raise NotImplementedError`）。
  3. 运行 `python makemore_part1_bigrams_exercise.py`，脚本会逐节做断言校验。

提示：整个练习只需要 torch，不需要 GPU，全部跑完不超过 1 分钟。
"""

import os

import torch
import torch.nn.functional as F

# 固定随机种子，保证结果可复现（课程里用的就是这个数）
SEED = 2147483647

# 词表大小：26 个字母 + 1 个特殊 token '.'（同时表示开始和结束）
VOCAB_SIZE = 27


# ---------------------------------------------------------------------------
# 0. 读数据
# ---------------------------------------------------------------------------

def load_words(path="names.txt"):
    """读入 names.txt，返回名字列表（每个元素是一个小写英文名，如 'emma'）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先下载数据集：\n"
            "  curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
        )
    return open(path, "r").read().splitlines()


# ---------------------------------------------------------------------------
# 1. 字符表：建立 字符 <-> 整数 的双向映射
# ---------------------------------------------------------------------------

def build_vocab(words):
    """
    构造 stoi / itos 两个字典。

    要求：
      - '.' 必须映射到 0（它既当开始符 <S> 也当结束符 <E>）；
      - 26 个字母按字典序映射到 1..26（'a' -> 1, 'b' -> 2, ..., 'z' -> 26）；
      - itos 是 stoi 的反向映射。

    返回: (stoi, itos)

    提示：先 set(''.join(words)) 拿到所有出现过的字符，sorted 之后 enumerate。
    """
    # 实现字符表
    chars = sorted(set(''.join(words)))
    stoi = {'.': 0}
    itos = {0: '.'}
    for i, char in enumerate(chars):
        stoi[char] = i + 1
        itos[i + 1] = char
    return stoi, itos


# ---------------------------------------------------------------------------
# 2. 统计法：数 bigram 次数，得到概率矩阵 P
# ---------------------------------------------------------------------------

def count_bigrams(words, stoi):
    """
    统计所有相邻字符对出现的次数，返回 shape 为 (27, 27) 的 int32 张量 N。
    N[i, j] 表示「字符 i 后面紧跟字符 j」出现了多少次。

    要求：
      - 每个单词前后都要补上 '.'，即 chs = ['.'] + list(w) + ['.']；
        这样才能统计到「哪些字符常做开头」和「哪些字符常做结尾」。
      - 用 zip(chs, chs[1:]) 遍历相邻字符对。

    提示：N = torch.zeros((27, 27), dtype=torch.int32)
    """
    # 实现 bigram 计数
    N = torch.zeros((27, 27), dtype=torch.int32)
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for (c1, c2) in zip(chs, chs[1:]):
            idx1 = stoi[c1]
            idx2 = stoi[c2]
            N[idx1, idx2] += 1
    return N


def normalize_counts(N, smoothing=1):
    """
    把计数矩阵 N 变成概率矩阵 P：每一行归一化，使得 P[i].sum() == 1。

    要求：
      - 先做加法平滑（model smoothing）：(N + smoothing)，
        避免出现次数为 0 的 bigram 概率为 0、取 log 后变成 -inf。
      - 转成 float，然后按「行」求和归一化。

    ⚠️ 广播陷阱（课程里重点讲的）：
        P.sum(1, keepdim=True) -> (27, 1)，与 (27, 27) 广播时按行对齐 ✅
        P.sum(1)               -> (27,)  == (1, 27)，广播时按列对齐 ❌
      所以必须写 keepdim=True。
    """
    # 实现归一化
    P = (N + smoothing).float()
    P = P / P.sum(1, keepdim=True)
    return P


def sample_from_P(P, itos, num=5, seed=SEED):
    """
    从概率矩阵 P 里采样 num 个名字，返回字符串列表。

    采样过程：
      - 从 ix = 0（也就是 '.'，开始符）出发；
      - 取出该行的概率分布 p = P[ix]，用 torch.multinomial 抽一个下标作为下一个字符；
      - 把字符追加到结果里，直到抽到 0（'.'，结束符）为止。

    注意：generator 要在循环外创建一次，这样 5 个名字共用一条随机数流。
          （torch 各版本 multinomial 的随机数实现有差异，采样出的具体名字不一定和
            课程视频里完全一致，不用纠结；只要看起来像"名字模样的乱码"就对了。）
    """
    # 实现采样
    g = torch.Generator().manual_seed(seed)
    name_list = []
    for i in range(num):
        ix = 0
        name = []
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g)
            if ix == 0:
                name_list.append(''.join(name))
                break
            name.append(itos[ix])
    return name_list

def nll_loss_from_P(words, P, stoi):
    """
    用概率矩阵 P 计算整个数据集的「平均负对数似然」（average negative log likelihood）。

    背景（课程里的推导）：
      目标 = 最大化数据的似然  prod(p_i)
           = 最大化 log 似然   sum(log p_i)     （log 单调递增）
           = 最小化负 log 似然 -sum(log p_i)
           = 最小化平均负 log 似然 -sum(log p_i) / n   <- 这就是我们的 loss

    返回一个 float。补全后在完整 names.txt 上应该约等于 2.4544。
    """
    # 实现 loss 计算
    return -torch.log(P).sum().item() / 27


# ---------------------------------------------------------------------------
# 3. 神经网络法：把 bigram 模型写成一层线性网络
# ---------------------------------------------------------------------------

def build_dataset(words, stoi):
    """
    把所有 bigram 摊平成训练集 (xs, ys) 两个 1D LongTensor。
      xs[k] = 第 k 个 bigram 的输入字符下标
      ys[k] = 第 k 个 bigram 的目标（下一个）字符下标

    同样每个单词要前后补 '.'。完整数据集上应该有 228146 个样本。
    """
    # TODO: 实现训练集构造
    raise NotImplementedError("build_dataset")


def forward(xs, W):
    """
    前向传播，返回 probs，shape 为 (len(xs), 27)。

    四步（每一步都对应课程里的一行）：
      1) xenc   = one-hot 编码输入，(N, 27) 的 float 张量。为什么要 float？
                  因为要送进矩阵乘法；one_hot 默认给的是 int64。
      2) logits = xenc @ W                      # 解释成 "log-counts"（对数计数）
      3) counts = logits.exp()                  # 取 exp 变回正数，等价于计数矩阵 N
      4) probs  = counts / counts.sum(1, keepdim=True)   # 按行归一化成概率

    第 3、4 两步合起来就是 softmax。

    小知识：one-hot 向量乘以矩阵 W，本质就是「取出 W 的第 ix 行」，
            所以这个网络和查表法在数学上是同一个东西。
    """
    # TODO: 实现前向传播
    raise NotImplementedError("forward")


def train(xs, ys, steps=100, lr=50.0, reg=0.01, seed=SEED, verbose=True):
    """
    用梯度下降训练权重 W，返回训练好的 W。

    步骤：
      - 初始化：W = torch.randn((27, 27), generator=g, requires_grad=True)
      - 循环 steps 次：
          forward pass：probs = forward(xs, W)
          loss = -probs[torch.arange(num), ys].log().mean() + reg * (W ** 2).mean()
                 ^ 取出每个样本「正确答案」那一格的概率，取 log、取负、求平均
                 ^ 后面那项是正则（regularization），作用等价于统计法里的加法平滑：
                   它把 W 往 0 拉，W 越接近 0，logits 越平，概率分布越均匀。
          backward pass：W.grad = None 然后 loss.backward()
                 ^ 注意先手动清零梯度，PyTorch 的梯度是累加的
          update：W.data += -lr * W.grad
                 ^ 沿负梯度方向走，lr=50 看着很大，但这个问题的 loss 面很平缓

    学习率 50、跑 100 步左右，loss 会收敛到 2.47 附近（略高于统计法，因为还没完全收敛）。
    """
    # TODO: 实现训练循环
    raise NotImplementedError("train")


def sample_from_W(W, itos, num=5, seed=SEED):
    """
    从训练好的网络里采样名字。

    和 sample_from_P 唯一的区别：
      统计法是   p = P[ix]
      神经网络是 先 one-hot(ix) -> @ W -> exp -> 归一化，得到 p
    其余（multinomial 采样、遇到 0 停止）完全一样。

    如果实现正确，输出应该和 sample_from_P 几乎一模一样 —— 这正是本节课的重点结论：
    两种方法学到的是同一个模型。
    """
    # TODO: 实现基于 W 的采样
    raise NotImplementedError("sample_from_W")


# ---------------------------------------------------------------------------
# 主流程：逐节自检
# ---------------------------------------------------------------------------

def main():
    words = load_words()
    print(f"数据集大小: {len(words)}，示例: {words[:5]}")
    print(f"最短名字长度: {min(len(w) for w in words)}，最长: {max(len(w) for w in words)}")

    # --- 1. 字符表 ---
    stoi, itos = build_vocab(words)
    assert len(stoi) == VOCAB_SIZE and stoi["."] == 0 and stoi["a"] == 1, "stoi 映射不对"
    assert itos[0] == "." and itos[26] == "z", "itos 映射不对"
    print("[1/6] 字符表 OK")

    # --- 2. bigram 计数 ---
    N = count_bigrams(words, stoi)
    assert N.shape == (VOCAB_SIZE, VOCAB_SIZE), "N 的形状应为 (27, 27)"
    # '.' 之后不可能立刻又是 '.'（没有空名字）
    assert N[0, 0].item() == 0, "N[0,0] 应为 0"
    # 训练集里以 'a' 开头的名字数量
    assert N[0, stoi["a"]].item() == 4410, f"N[., a] 期望 4410，实际 {N[0, stoi['a']].item()}"
    print("[2/6] bigram 计数 OK")

    # --- 3. 概率矩阵 + 统计法 loss ---
    P = normalize_counts(N)
    assert torch.allclose(P.sum(1), torch.ones(VOCAB_SIZE)), "P 每行之和应为 1"
    loss_count = nll_loss_from_P(words, P, stoi)
    print(f"      统计法 loss = {loss_count:.4f}")
    assert abs(loss_count - 2.4544) < 0.01, "统计法 loss 期望约 2.4544"
    print("[3/6] 概率矩阵与 loss OK")

    # --- 4. 统计法采样 ---
    names_p = sample_from_P(P, itos, num=5)
    print("      统计法采样:", names_p)
    print("[4/6] 统计法采样 OK")

    # --- 5. 神经网络训练 ---
    xs, ys = build_dataset(words, stoi)
    assert xs.shape == ys.shape and xs.nelement() == 228146, \
        f"训练样本数期望 228146，实际 {xs.nelement()}"
    W = train(xs, ys, steps=100, lr=50.0)
    probs = forward(xs, W)
    loss_nn = -probs[torch.arange(xs.nelement()), ys].log().mean().item()
    print(f"      神经网络 loss = {loss_nn:.4f}")
    assert loss_nn < 2.50, "训练 100 步后 loss 应低于 2.50"
    print("[5/6] 神经网络训练 OK")

    # --- 6. 神经网络采样，应与统计法结果一致 ---
    names_w = sample_from_W(W, itos, num=5)
    print("      神经网络采样:", names_w)
    print("[6/6] 神经网络采样 OK")

    print("\n全部通过 🎉  两种方法的 loss 差距:", f"{abs(loss_nn - loss_count):.4f}")


if __name__ == "__main__":
    main()
