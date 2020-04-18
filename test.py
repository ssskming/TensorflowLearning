import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('dot.csv')
x_date = np.array(df[['x1','x2']])
y_date = np.array(df['y_c'])

x_train = x_date
y_train = y_date.reshape(-1,1)

Y_c = [['red' if y else 'blue'] for y in y_train]

x_train = tf.cast(x_train,tf.float32)
y_train = tf.cast(y_train,tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)

w1 = tf.Variable(tf.random.normal([2,11]),dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01,shape=[11]))

w2 = tf.Variable(tf.random.normal([11,1]),dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01,shape=[1]))

lr = 0.01
epoch = 400

for epoch in range(epoch):
    for setp,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(y_train,w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1,w2) + b2

            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            loss_regularization = []

            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))

            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization

        variables = [w1,b1,w2,b2]
        grads = tape.gradient(loss,variables)

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

print('********************predict***************************')

xx,yy = np.mgrid[-3:3:0.1,-3:3:0.1]
grid = np.c_[xx.ravel(),yy.ravel()]
grid = tf.cast(grid,tf.float32)

probs = []
for x_predict in grid:
    h1 = tf.matmul([x_predict],w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1,w2) + b2
    probs.append(y)

x1 = x_date[:,0]
x2 = x_date[:,1]

probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1,x2,color=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[0.5])
plt.show()