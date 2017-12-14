
import numpy as np
import sys,os,glob
from keras import callbacks, optimizers
from deepModels import getModel
from deepModels import getSegModel
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import cv2

datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
batch_size = 16

model = getModel()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Prepare input data
classes = ['no','yes']

num_classes = len(classes)

# 10% of the data will automatically be used for validation
valsize = 0.05
img_size = 40
num_channels = 3
train_path='training/classification/'


## load up the images
X = []
y = []
np.random.seed(42)
for filename in os.listdir(train_path + '/yes/'):
    filename = train_path + "/yes/" + filename
    image = cv2.imread(filename)
    image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
    X.append(image)
    y.append(1)
for filename in os.listdir(train_path + '/no/'):
    filename = train_path + "/no/" + filename
    image = cv2.imread(filename)
    image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
    X.append(image)
    y.append(0)

X = np.asarray(X)
y = np.asarray(y)
X = X.astype('float32')/255

shuffle_index = np.random.permutation(X.shape[0])
X = X[shuffle_index]
y = y[shuffle_index]
y = to_categorical(y,2)

## use some for testing
val=int(valsize*len(X))
(X_train, y_train), (X_test, y_test) = (X[:val], y[:val]), (X[val:], y[val:])
print('Train set size : ', X_train.shape[0])
print('Test set size : ', X_test.shape[0])



# train the model
datagen = ImageDataGenerator(zoom_range=0.2, vertical_flip=False, horizontal_flip=True)
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128), steps_per_epoch = 512, nb_epoch = 100, validation_data = (X_test, y_test), verbose=1)

## save the weights
filepath='training/class_weights.hdf5'
model.save_weights(filepath)


