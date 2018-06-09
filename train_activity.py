import os
# from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.preprocessing.image import ImageDataGenerator

train_dir = '/home/kandithws/ait_workspace/MachineLearning/datasets/activity_dataset/train'
test_dir  = '/home/kandithws/ait_workspace/MachineLearning/datasets/activity_dataset/test'
saved_weights_name='best_weights.h5'

PREVIOUS_MODEL=None # path to previously trained model; for continue training

nb_train_samples = 200 # samples per epoch
nb_test_samples = 50
batch_size = 20
num_classes = 7
epochs = 30
width =300
height = 300

datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
training_generator = datagen.flow_from_directory(train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode="categorical")

testing_generator = datagen.flow_from_directory(test_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode="categorical")

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)

checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)
model = Sequential()
model.add(keras.applications.inception_v3.InceptionV3(include_top=True,
input_tensor=Input(shape=input_shape),
weights='imagenet',input_shape=input_shape, pooling=None, classes=1000))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

if PREVIOUS_MODEL is not None:
    model.load_weights(PREVIOUS_MODEL)
optimizer = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(training_generator,
          nb_epoch=epochs,
          verbose=1,
          callbacks  = [checkpoint, tensorboard],                     
                              steps_per_epoch=1,
          samples_per_epoch=nb_train_samples,
          nb_val_samples=nb_test_samples,
          validation_data=testing_generator)

