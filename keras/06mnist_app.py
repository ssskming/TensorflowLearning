from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist_f.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input("input the number of test piuctures:"))

for i in range(preNum):
    image_path = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC' + '\\' + input("the path of test picture:")
    #image_path = input("the path of test picture:")
    img = Image.open(image_path)
    img = img.resize((28,28),Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis,...]
    result = model.predict(x_predict)
    pred = tf.argmax(result,axis=1)
    print('\n')
    tf.print(pred)