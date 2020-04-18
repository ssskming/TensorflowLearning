# -*- coding: UTF-8 -*-

import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# tf.enable_resource_variables(
#     config = None
#     device_policy = None
#     execution_mode = None
# )

x_date = datasets.load_iris().data
y_date = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_date)
np.random.seed(116)
np.random.shuffle(y_date)
tf.random.set_random_seed(116)

x_train = x_date[:-30]
y_train = y_date[:-30]
x_test = x_date[-30:]
y_test = y_date[-30:]

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

# 超参数
lr = 0.1 
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1)
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train,depth=[3])
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()

        grands = tape.gradient(loss,[w1,b1])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print("Epoch: {},loss: {}".format(epoch,loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0

    total_correct,total_number = 0,0
    for x_test,y_test in test_db:
        y = tf.matmul(x_test,w1)
        y = tf.nn.softmax(y)

        pred = tf.argmax(y,axis=1)
        pred = tf.cast(pred,dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = int(total_correct)/int(total_number)
    test_acc.append[acc]
    print("Test_acc:",acc)
    print("----------------------------------------")

# 绘画 
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results,label="$Loss$")
plt.legend()
plt.show()

#绘制 Accuracy 
plt.title('Acc Curve')
plt.xlabel("Epoch")
plt.ylabel("acc")
plt.plot(test_acc,label="$Accuracy$")
plt.legend()
plt.show()

