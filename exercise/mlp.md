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

不幸的是train loss变高了，但是万幸的是dev loss-train loss降低了。猜测是由于context的增加，训练的learning rate和训练部署也要对应调整。





