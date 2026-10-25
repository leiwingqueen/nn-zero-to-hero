import torch

from exercise.makemore_part3_bn_exercise import init_mlp, train_loop, mlp_forward, load_words, build_vocab, split_words, \
    build_dataset, eval_loss
import matplotlib.pyplot as plt


# E01: 复现课程的 "loss log"。用 init_mlp 的四种 scheme 分别训练单隐藏层 MLP
#      （train_loop + mlp_forward，20 万步，lr_decay_at=0.5），记录 train/dev loss：
#          original 2.125/2.168 → fix softmax 2.07/2.13 → fix tanh 2.036/2.103
#          → kaiming 2.038/2.107 → batchnorm 2.067/2.105
#      体会：初始化修好以后，loss 曲线开头那段"曲棍球柄"消失了，省下的步数都用在了真正的学习上。

def main():
    words = load_words()
    stoi, itos = build_vocab(words)
    tr_words, dev_words, te_words = split_words(words)
    Xtr, Ytr = build_dataset(tr_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)
    Xte, Yte = build_dataset(te_words, stoi)
    print(f"数据集: train {tuple(Xtr.shape)}  dev {tuple(Xdev.shape)}  test {tuple(Xte.shape)}")

    fwd = lambda x: mlp_forward(x, params)[0]

    # --- 7. E01 ---
    for scheme in ["naive", "fix_softmax", "fix_tanh", "kaiming"]:
        params = init_mlp(scheme)
        steps = 200_000
        lossi = train_loop(Xtr, Ytr, params, lambda xb, p=params: mlp_forward(xb, p)[0], steps)
        plt.plot(torch.tensor(lossi).view(-1, 100).mean(1), label=f"scheme={scheme}")
        l_tr, l_dev = eval_loss(Xtr, Ytr, fwd), eval_loss(Xdev, Ydev, fwd)
        print(f"scheme:{scheme},train loss:{l_tr},dev loss:{l_dev}")
        # plt.plot(lossi)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
