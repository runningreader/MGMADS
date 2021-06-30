from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,BatchNormalization,SeparableConv2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import keras
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import ReduceLROnPlateau
import sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import random
from keras.metrics import top_k_categorical_accuracy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


model = ResNet50(include_top=True,
                 weights=None,
                 input_shape=(256, 256, 3),
                 classes=100)

model.summary()
adam= keras.optimizers.Adam(lr=0.0001)
model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=adam, metrics=['accuracy', acc_top5])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('/home/dataset/zhdy/miniimagenet/train',
                                                    target_size=(256,256),
                                                    batch_size=32,
                                                    )

validation_generator = test_datagen.flow_from_directory('/home/dataset/zhdy/miniimagenet/val',
                                                        target_size=(256,256),
                                                        batch_size=32,
                                                        )
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint =ModelCheckpoint(filepath='ImageNEtResNet_model.hdf5',
monitor='val_acc',
verbose=1,
save_best_only='True',
mode='max',
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
callback_list = [checkpoint, reduce_lr]
hist=model.fit_generator(train_generator,
                    steps_per_epoch=782,
                    epochs=50,
                    callbacks=callback_list,
                    validation_data=validation_generator,
                    validation_steps=157,
                    verbose=1)
model.save('ResNet_ImageNet_50epoch_model.hdf5')
model.save_weights('ResNet_ImageNet_weights.h5')
# def training_vis(hist):
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#     acc = hist.history['acc']
#     val_acc = hist.history['val_acc']
#
#     # make a figure
#     fig = plt.figure(figsize=(8,4))
#     # subplot loss
#     ax1 = fig.add_subplot(121)
#     ax1.plot(loss,label='train_loss')
#     ax1.plot(val_loss,label='val_loss')
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Loss on Training and Validation Data')
#     ax1.legend()
#     # subplot acc
#     ax2 = fig.add_subplot(122)
#     ax2.plot(acc,label='train_acc')
#     ax2.plot(val_acc,label='val_acc')
#     ax2.set_xlabel('Epochs')
#     ax2.set_ylabel('Accuracy')
#     ax2.set_title('Accuracy  on Training and Validation Data')
#     ax2.legend()
#     plt.tight_layout()
#     plt.savefig("ResNet_ImageNet.png")
# training_vis(hist)
# with open('VGG2lossacc.txt','w') as f:
#     f.write(str(hist.history))
# model.load_weights('3best_model.hdf5')
#
# score = model.evaluate(validation_generator)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])
# model.save('model2.hdf5')
#epoch=15 suit

