## hyper parameters train result


| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- |
| 3          | 10    | 200      | 100,000    | 2.1716     | 2.2054   |
| 3          | 10    | 200      | 200,000    | 2.1163     | 2.1654   |

train step由100,000增加到200,00，Train loss掉了0.05,dev loss掉了0.04，证明还在继续收敛

| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- |
| 3          | 10    | 200      | 300,000    | 2.0944     | 2.1520   |

尝试把step增加到300,000, train loss下降0.2，但是dev loss下降仅0.1，说明有overfiting的信号。继续增加步数已经没有太大意义。

接下来尝试增加N_EMB

| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- | ------------------- |
| 3          | 20    | 200      | 300,000    | 2.0482     | 2.1387   | 0.0905              |

dev loss-train loss增加到0.0905，overfiting的信号更明显了。

接下来尝试增加N_HIDDEN

| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- | ------------------- |
| 3          | 20    | 300      | 300,000    | 2.0144     | 2.1349   | 0.1206              |

接下来尝试调整block_size到5
| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- | ------------------- |
| 3          | 20    | 300      | 300,000    | 2.1260     | 2.1966   | 0.0705              |

不幸的是train loss变高了，但是万幸的是dev loss-train loss降低了。猜测是由于context的增加，训练的learning rate和训练步数对应调整。

最终调参得到的训练结果

| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- | ------------------- |
| 5          | 20    | 300      | 400,000    | 2.0905     | 2.1258   | 0.0353              |

## 修复batch_size后的结果

以上实验调用train时误把BLOCK_SIZE传给了batch_size，每步实际只用了3或5条样本。修复为batch_size=BATCH_SIZE后，用最终配置重跑：

| BLOCK_SIZE | N_EMB | N_HIDDEN | BATCH_SIZE | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | ---------- | -------- | ------------------- |
| 5          | 20    | 300      | 5（修复前） | 400,000    | 2.0905     | 2.1258   | 0.0353              |
| 5          | 20    | 300      | 64         | 300,000    | 1.7103     | 2.1021   | 0.3918              |

dev loss降低到2.1021，但是train loss降到1.7103，dev loss-train loss扩大到0.3918，脚本中`l_dev - l_tr < 0.15`的断言没有通过，overfiting非常明显。batch 64跑30万步相当于把训练集过了约100遍。

修复前gap一直很小，很可能是batch太小、梯度噪声大，起到了隐式正则的作用。之前的调参结论需要在batch 64下重新验证，下一步考虑减少步数、加weight decay或缩小网络来控制过拟合。首步loss为3.2926（缩放初始化），接近-ln(1/27)=3.2958。

E04（N_EMB=2，改回课程配置BLOCK_SIZE=3、N_HIDDEN=200、BATCH_SIZE=32、未缩放初始化，100,000步）：

| BLOCK_SIZE | N_EMB | N_HIDDEN | BATCH_SIZE | TRAIN_STEP | train loss | dev loss | dev loss-train loss |
| ---------- | ----- | -------- | ---------- | ---------- | ---------- | -------- | ------------------- |
| 3          | 2     | 200      | 32         | 100,000    | 2.2669     | 2.2703   | 0.0033              |

首步loss为23.0577，对比E01缩放初始化后的3.2926。







