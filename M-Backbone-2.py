import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import os

import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# y1=x
# y1 = Conv2D(256,(1,1),strides=2,activation='relu')(y1)
# y1 = Conv2D(256,(1,1),strides=2,activation='relu')(y1)
# y1 = Conv2D(64 * 3, (2, 2), activation='relu')(y1)
# if True:
#     y1 = Reshape([7 * 7, 64 * 3])(y1)
#     att = MultiHeadsAttModel(l=7 * 7, d=64 * 3, dv=8 * 3, dout=32, nv=8)
#     y1 = att([y1, y1, y1])
#     y1 = Reshape([7, 7, 32])(y1)
#     y1 = NormL()(y1)


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


y=layers.add([x,y2])
y = Activation('relu')(y)


x = GlobalAveragePooling2D()(y)
x = Dense(100, activation='softmax')(x)

model = Model(inputs=inp, outputs=x)

