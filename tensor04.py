# TensorFlow指定CPU和GPU设备操作详解

# log_device_plaement = true 可以指定设备运行
# allow_soft_plcement = true 自动选择受支持设备
import tensorflow as tf 
config = tf.ConfigProto(allow_soft_placement=0,log_device_placement=0)

# 手动选择cpu进行操作

with tf.device('/cpu:0'):
    rand_t = tf.random_uniform([50,50],0,10,dtype=tf.float32,seed=0)
    a = tf.Variable(rand_t)
    b = tf.Variable(rand_t)
    c = tf.matmul(a,b)
    init = tf.global_variables_initializer()
    
with  tf.Session(config) as sess:
    sess.run(init)
    print(sess.run(c))