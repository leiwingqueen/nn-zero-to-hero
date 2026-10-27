import matplotlib.pyplot as plt
import torch

from exercise.makemore_part3_bn_exercise import train_loop, load_words, build_vocab, split_words, \
    build_dataset, build_deep_net, deep_forward, set_training, SEED, N_EMB, BLOCK_SIZE, VOCAB_SIZE


# E02: 把 build_deep_net 的 use_bn 设成 False，分别用 gain = 0.5 / 1 / 5/3 / 3 训练几千步，
#      对比 loss 下降速度。再把 Tanh 全部去掉（只剩 Linear 堆叠）：
#        - 此时 gain=1 才是对的（为什么？提示：没有非线性来"压缩"分布）；
#        - 整个网络退化成了一个线性模型，loss 能降到多少？

def main():
    words = load_words()
    stoi, itos = build_vocab(words)
    tr_words, dev_words, te_words = split_words(words)
    Xtr, Ytr = build_dataset(tr_words, stoi)
    Xdev, Ydev = build_dataset(dev_words, stoi)
    Xte, Yte = build_dataset(te_words, stoi)
    print(f"数据集: train {tuple(Xtr.shape)}  dev {tuple(Xdev.shape)}  test {tuple(Xte.shape)}")

    gains = [0.5, 1, 5 / 3, 3]
    n_layer = 20
    for use_tanh in [True, False]:
        for gain in gains:
            C, layers, parameters = build_deep_net(n_layers=n_layer, use_bn=False, gain=gain, use_tanh=use_tanh)
            steps = 5_000
            print(f"gain:{gain}, 训练 {n_layer} 层 BN 网络 {steps} 步（CPU 约 20 秒）...")
            set_training(layers, True)
            lossi = train_loop(Xtr, Ytr, parameters, lambda xb: deep_forward(xb, C, layers), steps)
            # plt.plot(lossi)
            plt.plot(torch.tensor(lossi).view(-1, 100).mean(1), label=f"gain={gain:.2f},tanh={use_tanh}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
