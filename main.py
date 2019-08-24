import numpy as np

N = 0
data_x = np.array([[1,1,0.3],[1,0.4,0.5],[1,0.7,0.8]])
data_target = np.array([1,1,0])
data_w = np.array([0,0,0], dtype='f')

for i in data_x:
#    print(np.sum(i*data_w))
    data_w = (data_w + i) if (1 if np.sum(i*data_w) > data_target[N] else 0) == 0 \
        else ((data_w - i) if data_target[N] == 0 else data_w)

    N+=1
    print(data_w)

