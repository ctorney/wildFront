
import numpy as np
import sys,os,glob
from keras import callbacks, optimizers
from deepModels import getModel
from deepModels import getSegModel
from segGenerator import generate
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import cv2
import keras.backend as K

from keras.losses import categorical_crossentropy
import keras.backend as K

def w_categorical_crossentropy(y_true, y_pred):
    # y_true is a matrix of weight-hot vectors (like 1-hot, but they have weights instead of 1s)
    y_true_mask = K.clip(y_true, 0.0, 1.0)  # [0 0 W 0] -> [0 0 1 0] where W >= 1.
    cce = categorical_crossentropy(y_pred, y_true_mask)  # one dim less (each 1hot vector -> float number)
    cce = categorical_crossentropy(y_pred, y_true)  # one dim less (each 1hot vector -> float number)
    print(cce.shape)
    return cce
    y_true_weights_maxed = K.max(y_true, axis=-1)  # [0 120 0 0] -> 120 - get weight for each weight-hot vector
    wcce = cce * y_true_weights_maxed
    return K.sum(cce)

ny=256
nx=256
## load segmentation model
fcnmodel = getSegModel(ny,nx)

# load classification model
model = getModel()

# load save weights
filepath='training/class_weights.hdf5'
model.load_weights(filepath)

#transfer weights to FCN
flattened_layers = fcnmodel.layers
index = {}
for layer in flattened_layers:
    if layer.name:
        index[layer.name] = layer
for layer in model.layers:
    weights = layer.get_weights()
#    for i in range(len(weights)):
#        print(layer.name, ' ', i, ': ', weights[i].shape)
 #   if layer.name in ['fc1','fc2','predictions']:
 #       if layer.name == 'fc1':
 #           weights[0] = np.reshape(weights[0],(5,5,192,512))
 #       elif layer.name == 'fc2':
 #           weights[0] = np.reshape(weights[0],(1,1,512,256))
 #       else:
 #           weights[0] = np.reshape(weights[0],(1,1,256,2))
    if layer.name in index:
        print(layer.name)
        index[layer.name].set_weights(weights)

#for layer in fcnmodel.layers:
#    weights = layer.get_weights()
#    for i in range(len(weights)):
#        print(layer.name, ' ', i, ': ', weights[i].shape)
        
#sys.exit('bye!')
#
#datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
#batch_size = 16
#
#model = getModel()
#
fcnmodel.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])

#from keras.utils import plot_model
#plot_model( fcnmodel , show_shapes=True , to_file='model.png')

train_path = 'training/segmentation/'
train_batch_size = 128

datagen = generate(train_path, train_batch_size, nx, ny)


## save the weights
filepath='training/seg_weights.hdf5'

epochs = 200

class_weight = {0 : 1., 1: 100.}
for ep in range( epochs ):
    fcnmodel.fit_generator( datagen, steps_per_epoch = 512, nb_epoch = 1) #, class_weight=class_weight)
    fcnmodel.save_weights( filepath + "." + str( ep ) )
    		
fcnmodel.save_weights( filepath )
#
#
##Prepare input data
#classes = ['no','yes']
#
#num_classes = len(classes)
#
## 10% of the data will automatically be used for validation
#valsize = 0.05
#img_size = 40
#num_channels = 3
#train_path='training/classification/'
#
#
### load up the images
#X = []
#y = []
#np.random.seed(42)
#for filename in os.listdir(train_path + '/yes/'):
#    filename = train_path + "/yes/" + filename
#    image = cv2.imread(filename)
#    image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
#    X.append(image)
#    y.append(1)
#for filename in os.listdir(train_path + '/no/'):
#    filename = train_path + "/no/" + filename
#    image = cv2.imread(filename)
#    image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
#    X.append(image)
#    y.append(0)
#
#X = np.asarray(X)
#y = np.asarray(y)
#X = X.astype('float32')/255
#
#shuffle_index = np.random.permutation(X.shape[0])
#X = X[shuffle_index]
#y = y[shuffle_index]
#y = to_categorical(y,2)
#
### use some for testing
#val=int(valsize*len(X))
#(X_train, y_train), (X_test, y_test) = (X[:val], y[:val]), (X[val:], y[val:])
#print('Train set size : ', X_train.shape[0])
#print('Test set size : ', X_test.shape[0])
#
#
#
## train the model
#datagen = ImageDataGenerator(zoom_range=0.2, vertical_flip=False, horizontal_flip=True)
#model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128), steps_per_epoch = 512, nb_epoch = 100, validation_data = (X_test, y_test), verbose=1)
#
### save the weights
#
#model.save_weights(filepath)


