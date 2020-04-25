#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import multiprocessing.dummy
import tensorflow as tf
from tensorflow.keras 



model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.fit()