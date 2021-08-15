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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard

from keras.callbacks import ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
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
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(128, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = Conv2D(128, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(256, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(1, 1),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = Conv2D(512, kernel_size=(1, 1),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same')(x)
x = Conv2D(512, kernel_size=(1, 1),activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(64 * 3, (2, 2), activation='relu')(x)
if True:
    x = Reshape([7 * 7, 64 * 3])(x)
    att = MultiHeadsAttModel(l=7 * 7, d=64 * 3, dv=8 * 3, dout=32, nv=8)
    x = att([x, x, x])
    x = Reshape([7, 7, 32])(x)
    x = NormL()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(4, activation='softmax')(x)

model = Model(inputs=inp, outputs=x)
print(model.summary())

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=adam, metrics=['accuracy', acc_top5])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('D:/covid19_data/paperdata/train',
                                                    target_size=(256,256),
                                                    batch_size=8,
                                                    )

validation_generator = test_datagen.flow_from_directory('D:/covid19_data/paperdata/val',
                                                        target_size=(256,256),
                                                        batch_size=8,
                                                        )
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint =ModelCheckpoint(filepath='VAR1-Best_model.hdf5',
monitor='val_acc',
verbose=1,
save_best_only='True',
mode='max',
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
callback_list = [checkpoint, reduce_lr]
hist=model.fit_generator(train_generator,

                    epochs=50,
                    callbacks=callback_list,
                    validation_data=validation_generator,

                    verbose=1)
model.save('VAR1_epoch_model.hdf5')
model.save_weights('VAR1_weights.h5')



# model.load_weights('best_model.hdf5')
#
# score = model.evaluate(validation_generator)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

#epoch=15 suit

