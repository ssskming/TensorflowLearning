import tensorflow as tf 
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator #用于数据增强
from matplotlib import pyplot as plt

# 输出参数时，输出所有，而不用...代替
np.set_printoptions(threshold=np.inf)

# 数据集
train_path = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\fashion_train_jpg_60000'
train_txt = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\fashion_train_jpg_60000.txt'
x_train_savepath = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\mnist_x_train.npy'
y_train_savepath = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\mnist_y_train.npy'

test_path = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\fashion_test_jpg_10000'
test_txt = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\fashion_test_jpg_10000.txt'
x_test_savepath = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\mnist_x_test.npy'
y_test_savepath = r'E:\study\machine\bilibili4小时tensorflow课程源码\Mooc_tf2.0\class4\FASHION_FC\fashion_image_label\mnist_y_test.npy'

#生成自制数据集
def generateds(path,txt):
    f = open(txt,'r')  #以只读形式打开txt文件
    contents = f.readlines() # 读取文件中所有行
    f.close() # 关闭txt文件
    x,y_ = [],[] # 建立空列表 
    for content in contents:   # 逐行取出
        value = content.split() # 以空格分开，图片路经为value[0],标签文件为value[1],存入列表 
        img_path = path + '\\' + value[0] # 拉出图片路经和文件名
        img = Image.open(img_path) 
        img = np.array(img.convert('L')) # 图片变为8位宽灰度值的np.array格式 
        img = img / 255. # 数据归一化，（预处理）
        # img1.astype(img.dtype)
        x.append(img)  # 数据归一化的数据加入到列表 x
        y_.append(value[1]) # 标签加入到y_
        print('loading:' + content) # 打印状态提示
        
    x = np.array(x) # 变为np.array格式
    y_ = np.array(y_) # 变为np.array格式
    y_ = y_.astype(np.int64)
    return x,y_ #返回输入特征，返回标签Y

# train 判断是否有数据集，没有就自行制作
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------------------Load Datasets------------------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save,(len(x_train_save),28,28)) # reshpae
    x_test = np.reshape(x_test_save,(len(x_test_save),28,28))
else:
    print('--------------------------Generate Datasets-------------------')
    x_train,y_train = generateds(train_path,train_txt)
    x_test,y_test = generateds(test_path,test_txt)

    print('--------------------------Save Datasets------------------------')
    x_train_save = np.reshape(x_train,(len(x_train),-1))
    x_test_save = np.reshape(x_test,(len(x_test),-1))
    np.save(x_train_savepath,x_train_save)
    np.save(y_train_savepath,y_train)
    np.save(x_test_savepath,x_test_save)
    np.save(y_test_savepath,y_test)

# 进行数据增强
x_train = x_train.reshape(x_train.shape[0],28,28,1) # 给数据增加一个维度，使数据和网络结构匹配
image_gen_train = ImageDataGenerator(
    rescale=1./1., # 如为图像，分母为255时，可归至0-1
    rotation_range=45, # 随机45度旋转
    width_shift_range=.15, # 宽度偏移
    height_shift_range = 0.15, # 高度偏移
    horizontal_flip=True, # 水平翻转
    zoom_range=0.5 # 将图像随机缩放阀量50%
)
image_gen_train.fit(x_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# 用于保存模型参数
checkpoint_save_path = "./checkpoint/mnist_f.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------------load the model----------------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['sparse_categorical_accuracy'])

#model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
#model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),epochs=5,validation_data=(x_test,y_test),validation_freq=1)
history = model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),
                    epochs=50,
                    validation_data=(x_test,y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

################ save variables #########################
#print(model.trainable_variables)
file = open('./weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

################ show loss and acc#######################
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation_Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation_Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()