

import numpy as np
import random

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard



# ## Build the model
# 基础模型
def create_base_network(input_shape):
    model = Sequential()
    model.add(Dense(units=128, input_shape=(input_shape, ), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=128, activation='relu'))
    
    return model

# 计算欧式距离
def euclidean_distance(vects):
    v1, v2 = vects
    return K.sqrt(K.sum(K.square(v1 -v2), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    # 在这里我们需要求修改output_shape, 为(batch, 1)
    shape1, shape2 = shapes
    return (shape1[0], 1)

# 创建contrastive_loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    # 以0.5为阈值
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# 因为网络是一对样本作为输入的，可以为一堆正样本，或一对负样本
def create_pairs(x, digit_indices):
    pairs = []   # 用于存储每一对样本
    labels = []  # 用于表示是否为正样本或负样本，1为相同类别，0为不同类别
    
    n = min([len(digit_indices[d]) for d in range(10)]) - 1  # 用于获取类别中最少正样本的数目，以保证数据均衡
    
    for d in range(num_classes):
        for i in range(n):
            # 正样本的下标
            p1, p2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[p1], x[p2]]]
            # 产生1-9的随机整数，即选择负样本
            inc = random.randrange(1, 10)
            # 防止获取到同一类的数据
            dn = (d + inc) % 10
            n1, n2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[n1], x[n2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


# ## Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255.
X_test /= 255.


input_shape = X_train.shape[1:][0]
print("Input_Shape: ", input_shape)


num_classes = 10   # 类别总数


# ### 创建正样本positive samples和负样本negative samples
# 即从0-9，digit_indices[0]里全部数字0的正样本的下标，digit_indices[1]里则全为数字1的正样本的下标，后面的以此类推
digit_indices_train = [np.where(y_train == i)[0] for i in range(num_classes)]
digit_indices_test = [np.where(y_test == i)[0] for i in range(num_classes)]
# 生成pairs
tr_pairs, tr_y = create_pairs(X_train, digit_indices_train)
te_pairs, te_y = create_pairs(X_test, digit_indices_test)


# check, 每一个样本都包括了两个图像，
print("trainSet shape: ", tr_pairs.shape)
print("traingLabel shape: ", tr_y.shape)
print("testSet shape: ", te_pairs.shape)
print("testLabel shape: ", te_y.shape)


base_network = create_base_network(input_shape)

# 因为模型是以两个张量作为输入，然后将它们连接在以上的base_network，再输出一个结果
input_a = Input(shape=(input_shape, ))
input_b = Input(shape=(input_shape, ))

# 获取经过模型后的输出
processed_a = base_network(input_a)  
processed_b = base_network(input_b)


# 这里在创建一个Lambda层,用于计算base_network输出的两个特征的欧氏距离，并且不含有可训练参数的计算要求
distance = Lambda(function=euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])


model = Model(inputs=[input_a, input_b], outputs=distance)


rmsprop = RMSprop()
model.compile(optimizer=rmsprop, loss=contrastive_loss, metrics=[accuracy])
model.fit(x=[tr_pairs[:, 0], tr_pairs[:, 1]], y=tr_y, batch_size=128, epochs=20)

y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))



