from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten


def LeNet():
    '''
    这个LeNet-5版本可能与LeCun教授的论文不那么一致，如activation等，但都是遵从了大概的结构模型
    '''
    LeNet_5 = Sequential()
    LeNet_5.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', strides=1, input_shape=(28, 28, 1)))
    LeNet_5.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    LeNet_5.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', strides=1))
    LeNet_5.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    LeNet_5.add(Flatten())
    LeNet_5.add(Dense(units=120, activation='relu'))
    LeNet_5.add(Dense(units=84, activation='relu'))
    LeNet_5.add(Dense(units=10, activation='softmax'))
    
    return LeNet_5