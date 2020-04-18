import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
X = [1, 2]
state = [0.0, 0.0]
# 定义不同输入部分的权重
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])
# 定义输出层的权重
w_output = np.asarray([[0.1], [0.2]])
b_output = 0.1
# 按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i]*w_cell_input+b_cell
    state = np.tanh(before_activation)
    # 计算当前时刻的最终输出
    final_output = np.dot(state, w_output) + b_output
    # 输出每一时刻的信息
    print("before_activation", before_activation)
    print("state", state)
    print("final_output", final_output)