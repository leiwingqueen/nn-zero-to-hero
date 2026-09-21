## hyper parameters train result


| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- |
| 3          | 10    | 200      | 100,000    | 2.1716     | 2.2054   |
| 3          | 10    | 200      | 200,000    | 2.1163     | 2.1654   |

train step由100,000增加到200,00，Train loss掉了0.05,dev loss掉了0.04，证明还在继续收敛

| BLOCK_SIZE | N_EMB | N_HIDDEN | TRAIN_STEP | train loss | dev loss |
| ---------- | ----- | -------- | ---------- | ---------- | -------- |
| 3          | 10    | 200      | 300,000    | 2.0944     | 2.1520   |

尝试把step增加到300,000, train loss下降0.2，但是dev loss下降仅0.1，说明已经开始出现overfiting。继续增加步数已经没有太大意义。

