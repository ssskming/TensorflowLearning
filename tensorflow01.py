import tensorflow as tf
import os
message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
with tf.Session() as sess:
    print(sess.run(message).decode())
