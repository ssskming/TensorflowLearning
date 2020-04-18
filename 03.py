
import tensorflow as tf
from tensorflow import contrib
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
sample = "这是一个基于tensorflow的RNN短句子练习"
# 去重放入列表中
idx2char = list(set(sample))
print("idx2char", idx2char)
# 转换为字典，其中把字母作键，索引作为值
char2idx = {c: i for i, c in enumerate(idx2char)}
# 在字典里取出对应值，因为在idx2char中原sample句子的顺序已经被打乱
sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
 
# 设置该模型的一些参数
dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
 
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
# 将input转化为one-hot类型数据输出，此时X的维度变为[None, sequence_length, num_classes]
X_one_hot = tf.one_hot(X, num_classes)
 
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
# 加一层全连接层，相当于加一层深度，使预测更准确
outputs = contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=None)
print("outputs", tf.shape(outputs))
weights = tf.ones([batch_size, sequence_length])
# 此处包装了encoder和decoder
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
 
prediction = tf.argmax(outputs, axis=2)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", "".join(result_str))
    print(len(result_str))