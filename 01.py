import pickle
import numpy as np
import os
CIFAR_DIR = "E:\\study\\data\\cifar-10-python.tar\\cifar-10-python\\cifar-10-batches-py"
print (os.listdir(CIFAR_DIR))
print ("test")
with open(os.path.join(CIFAR_DIR,"data_batch_1"),'rb') as f:
    data = pickle.load(f)
    print (type(data))