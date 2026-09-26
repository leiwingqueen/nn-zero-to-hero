# 《A Neural Probabilistic Language Model》中文导读

> 原文：Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. *A Neural Probabilistic Language Model*. Journal of Machine Learning Research 3 (2003) 1137–1155.
> 原文链接：https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
>
> 说明：本文不是逐句翻译，而是按原文章节结构用中文重新组织的精读笔记，公式与符号尽量和原文保持一致，方便对照原文阅读。第 2.1 节补充了公式逐项推导，第 4 节补充了实验表格与逐项解读，文末附有关键原句的中英对照。它也是 Karpathy《makemore Part 2: MLP》所复现的那篇论文。

---

## 摘要（要点）

- 统计语言模型的目标是学习词序列的联合概率分布，最大的困难是**维度灾难**：测试时遇到的词序列很可能从未在训练集中出现过。
- 传统 n-gram 方法靠拼接训练集中出现过的短片段来泛化，能力有限。
- 本文提出：**为每个词学习一个分布式表示（词特征向量）**，并**同时**学习基于这些向量表示的词序列概率函数。
- 这样模型可以把一个句子的信息"传递"给大量语义相近的句子，从而指数级地扩大泛化范围。
- 实验表明，该神经网络模型在两个语料上都显著优于当时最好的 n-gram 模型，并且能利用更长的上下文。

---

## 1. 引言

### 维度灾难

以词表大小 |V| = 100,000 为例，建模 10 个连续词的联合分布，潜在自由参数量级为 100000¹⁰ − 1。离散变量上的任何一点变化都可能让函数值剧变，因此"见过的序列"几乎无法直接覆盖"没见过的序列"。

语言模型通常分解为条件概率连乘：

$$\hat P(w_1^T) = \prod_{t=1}^{T} \hat P(w_t \mid w_1^{t-1})$$

n-gram 进一步做马尔可夫近似，只看前 n−1 个词：

$$\hat P(w_t \mid w_1^{t-1}) \approx \hat P(w_t \mid w_{t-n+1}^{t-1})$$

对训练中没出现过的 n 元组，n-gram 依赖**回退（back-off）**或**插值平滑**（如 Katz、Kneser-Ney、删除插值），退到更短的上下文。作者指出它的两个缺陷：

1. 实际只能用到约 1–2 个词的上下文（三元模型），更长的上下文统计上用不起来；
2. 没有利用**词之间的相似性**。例如训练集中见过 "The cat is walking in the bedroom"，模型应该能推广到 "A dog was running in a room"，因为 cat/dog、the/a、walking/running、bedroom/room 在语义和语法上相近。

### 1.1 用分布式表示对抗维度灾难

核心思路分三步：

1. 给词表中每个词关联一个**分布式词特征向量**（实数向量，维度 m 远小于 |V|，例如 30、60、100）；
2. 用这些特征向量来表达词序列的**联合概率函数**；
3. **同时学习**词特征向量和概率函数的参数。

之所以能泛化：概率函数是特征向量的**光滑函数**，相似的词会得到相近的向量，因此特征的微小变化只引起概率的微小变化。一个训练句子不仅提高了它自身的概率，也提高了它在向量空间中"邻居句子"的概率。

### 1.2 与已有工作的关系

- 用神经网络建模高维离散分布的想法此前已有（如 Bengio & Bengio 2000 对联合概率的分解建模）。
- 分布式表示的思想可追溯到 Hinton (1986) 的连接主义工作；Elman、Miikkulainen & Dyer 等也用神经网络学习过词的表示。
- Xu & Rudnicky (2000) 用神经网络做过语言模型，但没有隐藏层、只用单个前一词作为输入。
- 信息检索中的 LSA（潜在语义分析）也学习词向量，但基于文档内共现，而本文基于**局部上下文的词序概率**。
- 与基于词类（word classes）的 n-gram 相比：词类是硬聚类，本文是连续的、多维的相似度。

---

## 2. 神经网络模型

训练集是词序列 w₁…w_T，w_t ∈ V。目标是学到一个好的模型：

$$f(w_t, \dots, w_{t-n+1}) = \hat P(w_t \mid w_1^{t-1})$$

唯一约束是：对任意上下文，所有 i 的 f(i, w_{t-1}, …, w_{t-n+1}) 之和为 1，且都 > 0。

模型把 f 拆成两部分：

1. **映射矩阵 C**：大小为 |V| × m，C(i) ∈ ℝᵐ 是词 i 的特征向量（即今天说的 embedding 查表）；
2. **概率函数 g**：把上下文词的特征向量序列 (C(w_{t-n+1}), …, C(w_{t-1})) 映射为下一个词在 V 上的条件分布，g 的第 i 个输出估计 $\hat P(w_t = i \mid w_1^{t-1})$：

$$f(i, w_{t-1}, \dots, w_{t-n+1}) = g\big(i, C(w_{t-1}), \dots, C(w_{t-n+1})\big)$$

C 在所有上下文位置间**共享**。

### 网络结构

把上下文词向量拼接成：

$$x = \big(C(w_{t-1}), C(w_{t-2}), \dots, C(w_{t-n+1})\big), \quad x \in \mathbb{R}^{(n-1)m}$$

未归一化的对数概率（logits）：

$$y = b + Wx + U \tanh(d + Hx)$$

再经 softmax 得到概率：

$$\hat P(w_t \mid w_{t-1}, \dots, w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}$$

各参数含义：

| 参数 | 形状 | 含义 |
|---|---|---|
| C | \|V\| × m | 词特征向量（embedding） |
| H | h × (n−1)m | 输入层 → 隐藏层权重 |
| d | h | 隐藏层偏置 |
| U | \|V\| × h | 隐藏层 → 输出层权重 |
| b | \|V\| | 输出层偏置 |
| W | \|V\| × (n−1)m | 输入层 → 输出层的**直连**权重（可选，不用时置 0） |

总参数 θ = (b, d, W, U, H, C)，参数量约为

$$|V|(1 + nm + h) + h\big(1 + (n-1)m\big)$$

主导项是 |V|(nm + h)，即**参数量与词表大小线性相关**，与上下文长度 n 也只是线性关系——这与 n-gram 随 n 指数增长形成鲜明对比。

### 训练目标

最大化训练语料上的带正则的平均对数似然：

$$L = \frac{1}{T}\sum_t \log f(w_t, w_{t-1}, \dots, w_{t-n+1}; \theta) + R(\theta)$$

R(θ) 是权重衰减，只作用于权重（H、U、W 和 C），不作用于偏置。

用**随机梯度上升**更新：

$$\theta \leftarrow \theta + \varepsilon \frac{\partial \log \hat P(w_t \mid w_{t-1}, \dots, w_{t-n+1})}{\partial \theta}$$

注意：每个样本只会更新**出现在上下文里的那几行 C**，其他词的向量不需要动。

### 与 n-gram 混合

实验中还把神经网络的输出与插值三元模型的输出**做概率平均**（混合），两者的错误模式不同，混合后效果更好。原文给出三种混合权重：固定 0.5；在验证集上用最大似然学一个权重；或按上下文频率分桶、每桶一组权重（与插值三元模型内部的做法相同）。表 1、表 2 中的 mix 一列用的是固定 0.5。

### 2.1 公式逐项推导

下面用单个样本 (w_{t-n+1}, …, w_{t-1}) → w_t 把前向、反向完整走一遍。记号：上下文长度 n−1，词向量维度 m，隐藏单元数 h，词表大小 |V|。

#### (1) 查表并拼接：C → x

$$x^{(k)} = C(w_{t-k}) \in \mathbb{R}^m,\quad k = 1,\dots,n-1$$

$$x = \big(x^{(1)}, x^{(2)}, \dots, x^{(n-1)}\big) \in \mathbb{R}^{(n-1)m}$$

这一层**没有非线性**。原文的解释是加了也没用：C(i) 本身就是自由参数，再套一个非线性，可以被 C 本身的取值吸收掉。

查表也可以看成矩阵乘法：one-hot 向量 $e_{w} \in \mathbb{R}^{|V|}$ 乘以 C，即 $C(w) = e_w^\top C$。所以 C 等价于一个**没有偏置、没有激活函数的线性层**，每个上下文位置都**共享**这一层。

#### (2) 隐藏层：x → a

$$o = d + Hx \in \mathbb{R}^h,\qquad a = \tanh(o)$$

H 的形状是 h × (n−1)m。可以把 H 按列切成 n−1 块：$H = [H^{(1)}, \dots, H^{(n-1)}]$，于是

$$Hx = \sum_{k=1}^{n-1} H^{(k)} x^{(k)}$$

也就是说：**不同位置的词向量是一样的（C 共享），但每个位置乘的是不同的 H 块**，所以模型能区分"前一个词是 cat"和"前两个词是 cat"。

#### (3) 输出层：得到 logits y

$$y = b + Wx + Ua \in \mathbb{R}^{|V|}$$

第 j 个分量写开是 $y_j = b_j + W_j \cdot x + U_j \cdot a$（$W_j$、$U_j$ 是 W、U 的第 j 行）。

- $U_j \cdot a$：非线性路径，经过隐藏层；
- $W_j \cdot x$：**直连路径**，从词向量直接到输出，是纯线性的。W = 0 时模型就是标准的单隐层 MLP。

#### (4) softmax 与损失

$$p_j = \frac{e^{y_j}}{\sum_{i=1}^{|V|} e^{y_i}},\qquad \mathcal{L} = \log p_{w_t}$$

论文**最大化**对数似然 $\mathcal{L}$，做梯度**上升**；PyTorch 里通常**最小化** $-\mathcal{L}$（即 `cross_entropy`），做梯度下降。两者等价，只差一个符号。

**数值稳定**：直接算 $e^{y_j}$ 可能上溢，也可能全部下溢成 0。原文的处理是先减去最大值 $Q = \max_j y_j$：

$$p_j = \frac{e^{y_j - Q}}{\sum_i e^{y_i - Q}}$$

分子分母同乘 $e^{-Q}$，结果不变。这样指数的最大值是 $e^0 = 1$，至少有一个 $p_j$ 不为 0。PyTorch 的 `F.cross_entropy` 内部也是这样做的。

#### (5) 反向：softmax + log 的梯度

$$\log p_{w_t} = y_{w_t} - \log \sum_i e^{y_i}$$

对 $y_j$ 求导：

$$\frac{\partial \mathcal{L}}{\partial y_j} = \mathbb{1}[j = w_t] - \frac{e^{y_j}}{\sum_i e^{y_i}} = \mathbb{1}[j = w_t] - p_j$$

这就是原文并行算法里的那一步 `∂L/∂y_j ← 1_{j==w_t} − p_j`。直观理解：把正确词的 logit 往上推（推力 1 − p），把其他词的 logit 按各自的概率往下压。

#### (6) 反向：输出层参数

记 $g = \partial \mathcal{L} / \partial y \in \mathbb{R}^{|V|}$：

$$\frac{\partial \mathcal{L}}{\partial b} = g,\qquad \frac{\partial \mathcal{L}}{\partial U} = g\, a^\top,\qquad \frac{\partial \mathcal{L}}{\partial W} = g\, x^\top$$

同时往下传：

$$\frac{\partial \mathcal{L}}{\partial a} = U^\top g,\qquad \frac{\partial \mathcal{L}}{\partial x}\Big|_{\text{直连}} = W^\top g$$

#### (7) 反向：穿过 tanh

$\tanh'(o) = 1 - \tanh^2(o) = 1 - a^2$，所以

$$\frac{\partial \mathcal{L}}{\partial o_k} = (1 - a_k^2)\, \frac{\partial \mathcal{L}}{\partial a_k}$$

$$\frac{\partial \mathcal{L}}{\partial d} = \frac{\partial \mathcal{L}}{\partial o},\qquad \frac{\partial \mathcal{L}}{\partial H} = \frac{\partial \mathcal{L}}{\partial o}\, x^\top,\qquad \frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= H^\top \frac{\partial \mathcal{L}}{\partial o}$$

注意 $\partial\mathcal{L}/\partial x$ 是**两条路径之和**：直连路径（W）加上隐藏层路径（H）。

这里也能看到 tanh 饱和的问题：若 $|o_k|$ 很大，$a_k \approx \pm 1$，$1 - a_k^2 \approx 0$，梯度就传不下去了。这正是 makemore Part 3 讨论初始化时关心的现象。

#### (8) 反向：更新词向量 C

把 $\partial \mathcal{L}/\partial x$ 按长度 m 切成 n−1 块，第 k 块加到对应词的那一行上：

$$C(w_{t-k}) \leftarrow C(w_{t-k}) + \varepsilon\, \frac{\partial \mathcal{L}}{\partial x^{(k)}}$$

- **稀疏更新**：一个样本只改 C 的 n−1 行，其余 |V| − (n−1) 行不动；
- 如果同一个词在窗口里出现多次，它那一行会**累加**多份梯度（PyTorch 里 `C[X]` 的反向就是 scatter-add）。

#### (9) 参数量怎么来的

按"每个输出词占多少参数"加"与词表无关的参数"拆开：

| 参数 | 数量 | 与 \|V\| 相关？ |
|---|---|---|
| b | \|V\| | 是 |
| C | \|V\| · m | 是 |
| U | \|V\| · h | 是 |
| W | \|V\| · (n−1)m | 是 |
| H | h · (n−1)m | 否 |
| d | h | 否 |

与 |V| 相关的部分：|V| · (1 + m + h + (n−1)m) = **|V|(1 + nm + h)**（这里的 nm 是 C 的 m 加上 W 的 (n−1)m 合起来的）。与 |V| 无关的部分：**h(1 + (n−1)m)**。这就是原文公式的来历。

代入表 1、表 2 里的几组配置（我自己算的，原文没有给出这些数）：

| 配置 | \|V\| | n | m | h | 直连 | 总参数 | 其中 W 占 |
|---|---|---|---|---|---|---|---|
| Brown MLP1 | 16,383 | 5 | 60 | 50 | 有 | ≈ 576 万 | ≈ 393 万 |
| Brown MLP9 | 16,383 | 5 | 30 | 100 | 无 | ≈ 216 万 | 0 |
| AP News MLP10 | 17,964 | 6 | 100 | 60 | 有 | ≈ 1190 万 | ≈ 898 万 |

可以看到**直连 W 往往是参数量的大头**，因为它的形状是 |V| × (n−1)m。

**关于 C 发散**：原文提到一个理论上的隐患：权重衰减如果只作用在 W、H 上而不作用在 C 上，模型可以把 W、H 缩小、同时把 C 放大，输出不变但惩罚变小，最终 C 发散。作者说实际用随机梯度训练时没有观察到这种现象；他们的做法是 C 也加权重衰减。

#### (10) 计算量：为什么瓶颈在输出层

处理一个样本的乘加次数大约是（原文近似）：

$$\underbrace{|V|(1 + (n-1)m + h)}_{\text{输出层}} + \underbrace{h(1 + (n-1)m)}_{\text{隐藏层}} + \underbrace{(n-1)m}_{\text{查表}}$$

以 AP News 的配置代入，输出层约占 **99.7%**。所以原文第 3 节的并行策略，以及后来的 hierarchical softmax、负采样，都是冲着输出层的 softmax 去的。

**对比 n-gram**：n-gram 查一个条件概率只要查表，不用对整个词表归一化（归一化在训练时统计频率就已经完成了）；神经网络每次预测都要算完 |V| 个 logits 才能归一化。

---

## 3. 并行实现

计算瓶颈在输出层：每个样本都要对全部 |V| 个输出单元算 logits 和 softmax。作者尝试了两种并行方式：

- **数据并行**（共享内存多处理器，用于 Brown 语料）：多个处理器各自处理不同样本，**异步**地直接更新共享参数，不加锁。虽然偶尔会有写冲突，但实验表明噪声可以接受，且速度提升明显。
- **参数并行**（由多台普通 PC 组成的集群，用于 AP News 语料）：把**输出单元按词表切分**到各个 CPU，每个 CPU 只负责一部分输出的 logits 与梯度。softmax 的归一化项通过各 CPU 之间交换部分和来完成，通信量很小，因此扩展性很好。

---

## 4. 实验

### 数据集

- **Brown 语料**：约 118 万词。前约 80 万词训练、接下来约 20 万词作为验证集（选模型、早停）、剩余约 18 万词测试。把出现次数很少的词合并为一个特殊符号后，词表约 1.6 万。
- **AP News（美联社新闻，1995–1996）**：训练集约 1400 万词，验证、测试各约 100 万词。做了大小写归一化并合并稀有词，词表约 1.8 万。

### 评价指标

**困惑度（perplexity）**：$\exp\big(-\frac{1}{T}\sum \log \hat P(w_t \mid w_1^{t-1})\big)$，即平均负对数似然的指数，越低越好。

### 对比基线

插值 / 回退 n-gram，包括删除插值（deleted interpolation）的三元模型、Kneser-Ney 平滑的回退模型，以及基于词类的 n-gram。

### 训练细节

| 项目 | 取值 |
|---|---|
| 初始学习率 ε₀ | 10⁻³（在很小的数据集上试了几次选出来的） |
| 学习率衰减 | $\varepsilon_t = \varepsilon_0 / (1 + r t)$，t 是已做的参数更新次数，r = 10⁻⁸ |
| 权重衰减 | Brown 用 10⁻⁴，AP News 用 10⁻⁵（按验证集困惑度选出来的） |
| 词向量初始化 | 随机初始化，和网络权重一样；作者怀疑用先验知识初始化会更好 |
| 早停 | 用了验证集早停，但只有 Brown 实验真正需要 |
| Brown 训练时长 | 约 10–20 个 epoch 看起来收敛 |
| AP News 训练时长 | 只跑了 5 个 epoch，**40 个 CPU 跑了约 3 周**，验证集上没有出现过拟合 |

r = 10⁻⁸ 衰减得很慢：Brown 训练集约 80 万词，一个 epoch 约 8×10⁵ 次更新，20 个 epoch 后 rt ≈ 0.16，学习率只降到约 0.86 ε₀。

### 4.1 n-gram 基线：插值三元模型长什么样

原文第一个基线是删除插值（deleted interpolation）三元模型，是一个**条件混合**：

$$\hat P(w_t \mid w_{t-1}, w_{t-2}) = \alpha_0(q_t)\,p_0 + \alpha_1(q_t)\,p_1(w_t) + \alpha_2(q_t)\,p_2(w_t \mid w_{t-1}) + \alpha_3(q_t)\,p_3(w_t \mid w_{t-1}, w_{t-2})$$

- $p_0 = 1/|V|$（均匀分布），$p_1$、$p_2$、$p_3$ 分别是一元、二元、三元的相对频率；
- $q_t$ 是上下文 $(w_{t-1}, w_{t-2})$ 出现频率离散化后的"桶号"，每个桶有一组权重 α，α ≥ 0 且和为 1；
- 思路：上下文很常见时相信 $p_3$；上下文罕见时更多地退回 $p_2$、$p_1$ 甚至 $p_0$；
- α 用 EM 在验证集上估计，约 5 轮就收敛。

其他基线用 SRILM 工具包实现：**修正 Kneser-Ney 回退 n-gram**，以及**基于词类的 n-gram**（Brown 等人的聚类方法）。n-gram 的阶数和词类数都在验证集上选。

为了公平比较，计算困惑度时所有 token（词和标点）一视同仁，句子结束符也不做特殊处理。

### 4.2 Brown 语料结果（原文表 1）

列含义：n 为模型阶数（上下文 n−1 个词）；c 为词类数（只对 class-based 模型）；h 为隐藏单元数；m 为词向量维度；direct 表示有没有直连 W；mix 表示是否与插值三元模型各取 0.5 做平均。

| 模型 | n | c | h | m | direct | mix | 训练 | 验证 | 测试 |
|---|---|---|---|---|---|---|---|---|---|
| MLP1 | 5 | | 50 | 60 | 有 | 否 | 182 | 284 | 268 |
| MLP2 | 5 | | 50 | 60 | 有 | 是 | | 275 | 257 |
| MLP3 | 5 | | 0 | 60 | 有 | 否 | 201 | 327 | 310 |
| MLP4 | 5 | | 0 | 60 | 有 | 是 | | 286 | 272 |
| MLP5 | 5 | | 50 | 30 | 有 | 否 | 209 | 296 | 279 |
| MLP6 | 5 | | 50 | 30 | 有 | 是 | | 273 | 259 |
| MLP7 | 3 | | 50 | 30 | 有 | 否 | 210 | 309 | 293 |
| MLP8 | 3 | | 50 | 30 | 有 | 是 | | 284 | 270 |
| MLP9 | 5 | | 100 | 30 | 无 | 否 | 175 | 280 | 276 |
| **MLP10** | 5 | | 100 | 30 | 无 | 是 | | **265** | **252** |
| 删除插值三元 | 3 | | | | | | 31 | 352 | 336 |
| Kneser-Ney 回退 | 3 | | | | | | | 334 | 323 |
| Kneser-Ney 回退 | 4 | | | | | | | 332 | 321 |
| Kneser-Ney 回退 | 5 | | | | | | | 332 | 321 |
| class-based 回退 | 3 | 150 | | | | | | 348 | 334 |
| class-based 回退 | 3 | 200 | | | | | | 354 | 340 |
| **class-based 回退** | 3 | 500 | | | | | | **326** | **312** |
| class-based 回退 | 3 | 1000 | | | | | | 335 | 319 |
| class-based 回退 | 3 | 2000 | | | | | | 343 | 326 |
| class-based 回退 | 4 | 500 | | | | | | 327 | 312 |
| class-based 回退 | 5 | 500 | | | | | | 327 | 312 |

> 注：删除插值三元模型那一行的训练集困惑度原文就是 31。它直接用训练集频率估计，在训练集上几乎是"背答案"，这个数说明不了泛化能力。
>
> 最好的模型按**验证集**选，不按测试集选：神经网络选 MLP10（验证 265），n-gram 选 c=500 的 class-based 三元（验证 326）。

### 4.3 AP News 结果（原文表 2）

| 模型 | n | h | m | direct | mix | 验证 | 测试 |
|---|---|---|---|---|---|---|---|
| **MLP10** | 6 | 60 | 100 | 有 | 是 | **104** | **109** |
| 删除插值三元 | 3 | | | | | 126 | 132 |
| Kneser-Ney 回退 | 3 | | | | | 121 | 127 |
| Kneser-Ney 回退 | 4 | | | | | 113 | 119 |
| Kneser-Ney 回退 | 5 | | | | | 112 | 117 |

在这个大语料上，class-based 模型对 n-gram 没有帮助；n-gram 里最好的是高阶修正 Kneser-Ney。

### 4.4 逐项对照：各配置怎么影响困惑度

下面每组对比只改一个变量（除特别说明外，看**测试集**困惑度）。

**(a) 神经网络 vs 最好的 n-gram**

| 语料 | 神经网络 | 最好的 n-gram | 相对差距 |
|---|---|---|---|
| Brown | MLP10：252 | class-based 三元（c=500）：312 | 312 / 252 − 1 ≈ **24%** |
| Brown | MLP10：252 | 删除插值三元：336 | 336 / 252 − 1 ≈ **33%** |
| AP News | MLP10：109 | Kneser-Ney 五元：117 | 117 / 109 − 1 ≈ **7–8%** |

注意"24%"的算法是 n-gram 比神经网络**高出**多少（以神经网络为分母）。以 n-gram 为分母算"降低了多少"，是 (312 − 252) / 312 ≈ 19%。原文结论部分说"10% 到 20% 的差距"，用的大概是后一种算法，对比对象是平滑三元模型。

语料越大差距越小（24% → 8%），原因很好理解：数据多了，n-gram 的统计稀疏问题缓解了，神经网络靠词相似性泛化的优势就没那么突出。

**(b) 上下文长度 n：神经网络能用上更长的上下文**

| 对比 | n=3 | n=5 | 变化 |
|---|---|---|---|
| MLP7 → MLP5（不混合） | 293 | 279 | −14 |
| MLP8 → MLP6（混合） | 270 | 259 | −11 |
| Kneser-Ney | 323 | 321（n=4 也是 321） | −2 |
| class-based（c=500） | 312 | 312（n=4 也是 312） | 0 |

n-gram 超过三元后几乎不再提升（长上下文的组合在训练集里太稀疏）；神经网络从 2 个词的上下文加到 4 个词，还能明显下降。这是论文的核心卖点之一。

**(c) 隐藏层：h = 50 vs h = 0**

| 对比 | h=0 | h=50 | 变化 |
|---|---|---|---|
| MLP3 → MLP1（不混合） | 310 | 268 | −42 |
| MLP4 → MLP2（混合） | 272 | 257 | −15 |

h = 0 时模型只剩直连路径 y = b + Wx，对词向量是**纯线性**的（相当于一个"对数双线性"模型）。加上 tanh 隐藏层后提升很大，说明非线性组合上下文信息是有价值的。

**(d) 词向量维度 m：60 vs 30**

| 对比 | m=30 | m=60 | 变化 |
|---|---|---|---|
| MLP5 → MLP1（不混合） | 279 | 268 | −11 |
| MLP6 → MLP2（混合） | 259 | 257 | −2 |

维度加大有帮助，但混合三元模型之后差距几乎消失。这组对比原文没有单独讨论，是从表里读出来的。

**(e) 与三元模型混合：每一组都有效**

| 模型对 | 不混合 | 混合 | 变化 |
|---|---|---|---|
| MLP1 / MLP2 | 268 | 257 | −11 |
| MLP3 / MLP4 | 310 | 272 | −38 |
| MLP5 / MLP6 | 279 | 259 | −20 |
| MLP7 / MLP8 | 293 | 270 | −23 |
| MLP9 / MLP10 | 276 | 252 | −24 |

用**固定权重 0.5 简单平均**就能稳定提升，说明两个模型在**不同的地方犯错**：各自给某些实际出现的词分配了很低的概率，而这些地方基本不重叠。模型越弱（如 h=0 的 MLP3），从混合中获益越多。

**(f) 直连 W：结论不确定**

MLP9/MLP10 同时改了三个变量（h 从 50 到 100，m 从 60 到 30，去掉直连），所以不是严格的单变量对比。原文的解读比较谨慎：

- 数据不足以断定直连有没有用；
- 但至少在小语料上，**去掉直连泛化更好**，代价是训练更慢：没有直连要约 20 个 epoch 才收敛，有直连约 10 个，而前者最终困惑度略低；
- 解释：直连提供了额外容量，能快速学到词向量到 log 概率的"线性部分"；去掉直连后，隐藏层成了一个**窄瓶颈**，迫使模型学到更好泛化的表示。

**(g) 过拟合迹象**

Brown 上 MLP 的训练困惑度（175–210）明显低于验证困惑度（280–327），存在一定过拟合，所以需要早停和权重衰减。AP News 训练集大约大 17 倍，只跑了 5 个 epoch，验证集上没看到过拟合，说明模型在大语料上**还没训练饱和**，继续训练可能更好。

---

## 5. 扩展与未来工作

作者列出了一些改进方向（很多后来都成为了现实）：

1. **把网络拆成子网络**，例如按词聚类，让不同子网络负责不同的输出词簇，降低计算量；
2. **树结构 / 层次化输出**：把对 |V| 个词的 softmax 分解成沿树路径的一系列二分类，使计算量从 O(|V|) 降到 O(log|V|)（后来的 hierarchical softmax）；
3. **只对一部分输出单元计算梯度**（类似后来的负采样 / 重要性采样思想）；
4. **引入先验知识**：如语义信息（WordNet）、词性标注、形态学信息等；
5. **分析学到的词向量**：例如用降维可视化，观察语义相近的词是否聚在一起；
6. **一词多义**：当前模型每个词只有一个向量，可以为每个词分配多个特征向量来处理多义词；
7. **循环神经网络**：用 RNN 替代固定窗口，以捕捉更长的上下文；
8. **应用到语音识别、机器翻译等任务**中验证实际收益。

原文此处还讨论了两个变体：

- **能量最小化网络**：把"下一个词"也作为输入映射成特征向量，网络输出一个能量（标量），概率正比于 exp(−能量)。这种形式可以让输入和输出共享词向量。
- **处理未登录词（OOV）**：对一个新词，可以用模型在该上下文下预测出的候选词分布，对这些词的特征向量加权平均，作为新词的初始向量。

---

## 6. 结论

- 在两个语料上，该神经语言模型都显著优于当时最好的平滑 n-gram 模型。
- 优势来源于**学习到的分布式词表示**：它让模型可以在相似词、相似句子之间共享统计强度，从而对抗维度灾难。
- 主要代价是计算量大，尤其是输出层的 softmax；作者认为通过并行化和结构改进可以缓解。
- 这项工作开启了用神经网络做语言建模、学习词向量的方向（后来的 word2vec、RNNLM 乃至 Transformer 语言模型都沿着这条路发展）。

---

## 附：关键原句对照

从原文各节挑出几句最能概括思想的话，逐句给出中文翻译。括号里标注出处，方便回原文定位。

**1. 问题是什么（摘要）**

> This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training.

这件事本质上很难，原因是维度灾难：模型测试时遇到的词序列，很可能与训练中见过的所有词序列都不相同。

**2. 解决思路（摘要）**

> We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences.

我们提出通过学习词的分布式表示来对抗维度灾难：每个训练句子都能让模型了解到指数级数量的、语义上相邻的句子。

**3. 泛化从何而来（摘要）**

> Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence.

泛化之所以成立，是因为：一个从没见过的词序列，只要组成它的词与某个已见过句子中的词相似（即表示向量相近），就会得到较高的概率。

**4. n-gram 是怎么泛化的（第 1 节）**

> Essentially, a new sequence of words is generated by "gluing" very short and overlapping pieces of length 1, 2 ... or up to n words that have been seen frequently in the training data.

本质上，新的词序列是由训练数据中频繁出现的、长度为 1、2……直到 n 的相互重叠的短片段"粘"起来生成的。

**5. n-gram 的两个缺陷（第 1 节）**

> First, it is not taking into account contexts farther than 1 or 2 words, second it is not taking into account the "similarity" between words.

第一，它不考虑超过 1 到 2 个词之外的上下文；第二，它不考虑词与词之间的"相似性"。

**6. 为什么光滑性带来泛化（第 1.1 节）**

> ... because the probability function is a smooth function of these feature values, a small change in the features will induce a small change in the probability.

……因为概率函数是这些特征值的光滑函数，特征的微小变化只会引起概率的微小变化。

**7. 参数规模（第 2 节）**

> In the above model, the number of free parameters only scales linearly with V, the number of words in the vocabulary.

在上述模型中，自由参数的数量只随词表大小 V 线性增长。

**8. 稀疏更新（第 2 节）**

> Note that a large fraction of the parameters needs not be updated or visited after each example: the word features C(j) of all words j that do not occur in the input window.

注意，每处理一个样本后，很大一部分参数不需要更新甚至不需要访问：所有没有出现在输入窗口中的词 j，其词向量 C(j) 都不用动。

**9. 计算瓶颈（第 3 节）**

> The main computational bottleneck with the neural implementation is the computation of the activations of the output layer.

神经网络实现的主要计算瓶颈在于输出层激活值的计算。

**10. 为什么混合有效（第 4.2 节）**

> The fact that simple averaging helps suggests that the neural network and the trigram make errors (i.e. low probability given to an observed word) in different places.

简单平均就能带来提升，这说明神经网络和三元模型在不同的地方犯错（"犯错"指给实际出现的词分配了很低的概率）。

**11. 去掉直连的解释（第 4.2 节）**

> On the other hand, without those connections the hidden units form a tight bottleneck which might force better generalization.

另一方面，没有这些直连时，隐藏单元构成了一个狭窄的瓶颈，这可能迫使模型学到泛化更好的表示。

**12. 一词多义的局限（第 5.2 节）**

> Polysemous words are probably not well served by the model presented here, which assigns to each word a single point in a continuous semantic space.

本文的模型给每个词在连续语义空间中只分配一个点，因此对多义词大概处理得不好。

**13. 核心结论（第 6 节）**

> ... the proposed approach allows to take advantage of the learned distributed representation to fight the curse of dimensionality with its own weapons: each training sentence informs the model about a combinatorial number of other sentences.

……所提出的方法利用学到的分布式表示，"以其人之道还治其人之身"地对抗维度灾难：每个训练句子都能让模型了解到组合数量级的其他句子。

**14. 把难点转移到了别处（第 6 节）**

> ... many more computations are required, but computation and memory requirements scale linearly, not exponentially with the number of conditioning variables.

……需要多得多的计算，但计算量和内存需求随条件变量个数线性增长，而不是指数增长。

---

## 附：与本仓库 makemore MLP 代码的对应关系

| 论文符号 | makemore 代码中的对应 |
|---|---|
| C（\|V\| × m） | `C = torch.randn((vocab_size, n_embd))`，字符级时 \|V\| = 27 |
| n − 1（上下文长度） | `block_size` |
| x = 拼接的上下文向量 | `emb = C[X]; emb.view(-1, block_size * n_embd)` |
| H, d | `W1, b1` |
| tanh(d + Hx) | `h = torch.tanh(emb @ W1 + b1)` |
| U, b | `W2, b2` |
| W（直连） | 通常省略 |
| softmax + 负对数似然 | `F.cross_entropy(logits, Y)` |
| 学习率衰减、权重衰减 | 手动调整的 `lr`、可选的 L2 正则 |

区别在于：论文是**词级**模型（|V| 约 1.6–1.8 万），makemore 是**字符级**模型（|V| = 27），但网络结构是一样的。
