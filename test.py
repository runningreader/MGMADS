# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing import image
import time
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import time
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,BatchNormalization,SeparableConv2D,AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import os
from keras.layers import Layer
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras import backend as K
import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard
#import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import confusion_matrix
import itertools
from keras import layers
import sys
import codecs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


file_list = os.listdir('D:/covid19_data/paperdata/test')
images = []
def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
    v1 = Input(shape=(l, d))
    q1 = Input(shape=(l, d))
    k1 = Input(shape=(l, d))

    v2 = Dense(dv * nv, activation="relu")(v1)
    q2 = Dense(dv * nv, activation="relu")(q1)
    k2 = Dense(dv * nv, activation="relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)

    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q, k])  # l, nv, nv
    att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)

    out = Add()([out, q1])

    out = Dense(dout, activation="relu")(out)

    return Model(inputs=[q1, k1, v1], outputs=out)

class NormL(Layer):

    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=True)
        self.b = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out * self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape
def main():
    inp = Input(shape=(256, 256, 3))
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(inp)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(128, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(256, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(1, 1),activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    y1=x
    y1 = Conv2D(256,(1,1),strides=2,activation='relu')(y1)
    y1 = Conv2D(256,(1,1),strides=2,activation='relu')(y1)
    y1 = Conv2D(64 * 3, (2, 2), activation='relu')(y1)
    if True:
        y1 = Reshape([7 * 7, 64 * 3])(y1)
        att = MultiHeadsAttModel(l=7 * 7, d=64 * 3, dv=8 * 3, dout=32, nv=8)
        y1 = att([y1, y1, y1])
        y1 = Reshape([7, 7, 32])(y1)
        y1 = NormL()(y1)


    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(1, 1),activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    y2=x
    y2 = Conv2D(256,(1,1),strides=2,activation='relu')(y2)
    y2 = Conv2D(64 * 3, (2, 2), activation='relu')(y2)
    if True:
        y2 = Reshape([7 * 7, 64 * 3])(y2)
        att = MultiHeadsAttModel(l=7 * 7, d=64 * 3, dv=8 * 3, dout=32, nv=8)
        y2 = att([y2, y2, y2])
        y2 = Reshape([7, 7, 32])(y2)
        y2 = NormL()(y2)



    x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same')(x)
    x = Conv2D(512, kernel_size=(1, 1),activation='relu', padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(64 * 3, (2, 2), activation='relu')(x)
    if True:
        x = Reshape([7 * 7, 64 * 3])(x)
        att = MultiHeadsAttModel(l=7 * 7, d=64 * 3, dv=8 * 3, dout=32, nv=8)
        x = att([x, x, x])
        x = Reshape([7, 7, 32])(x)
        x = NormL()(x)


    y=layers.add([x,y1,y2])
    y = Activation('relu')(y)


    x = GlobalAveragePooling2D()(y)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    # model = Model(inputs=inp, outputs=x)
    model.load_weights('D:/covid_code/yongge/weights.h5')
    # model=load_model("ResNet2_jing_50epoch_model.hdf5")
    # model.load_weights('imp_attention_weights.h5')
    # model = load_model(r"C:\Users\aurora\Desktop\classificate/models/ImageNEtResNet_model.hdf5")
    right = []
    right_file = []
    error = []
    picture = []
    fail = []

    for file in file_list:

        img = image.load_img(os.path.join('D:/covid19_data/paperdata/test/', file), target_size=(256, 256))


        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    x_train = np.array(images, dtype="float") / 255.0
    x = np.concatenate([x for x in x_train])

    time_start = time.time()
    y = model.predict(x)
    time_end = time.time()
    j=0
    # names.append(file)
    # print(names)

    for i in range(len(file_list)):
        if np.argmax(y[i]) == 0:
            print('image {} class:'.format(file_list[i]), '0')

        elif np.argmax(y[i]) == 1:
            print('image {} class:'.format(file_list[i]), '1')

        elif np.argmax(y[i]) == 2:
            print('image {} class:'.format(file_list[i]), '2')

        elif np.argmax(y[i]) == 3:
            print('image {} class:'.format(file_list[i]), '3')

    for filename in file_list:
        name = filename.split('_')
        # print(name)
        # print(name[0][0])
        #
        # print(np.argmax(y[j]))
        # print(name)

        # print(name[0])
        if (int(name[0][0])) == np.argmax(y[j]):
                # print((int(name[0])))
                right.append(1)
                right_file.append(filename)
        else:
                error.append(0)
                fail.append(filename)
        j = j + 1

    print("right：", len(right), "error：", len(error))

    print("fail：", fail)

# starttime = datetime.datetime.now()
# endtime=datetime.datetime.now()
#     time_end = time.time()
    print('time:', (time_end - time_start))
    print(time_start)
    print(time_end)
    # print('time:',(starttime-endtime))
main()

