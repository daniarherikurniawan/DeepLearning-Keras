# cd ~/.keras/ sublime keras.json change to theano or tensorflow ONLY
# run jupyter using: jupyter notebook
# better to edit the json file!
import os
os.environ["OMP_NUM_THEADS"]="2000"
import theano
theano.config.openmp = True

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

from datetime import datetime, timedelta


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


import platform
 
print(platform.python_version())

import keras
keras.backend.backend()

dog_files_short = train_files[:100]

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
#     return np.rollaxis(y, 4, 1)  

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

#Obtain the bottleneck feature for Resnet50 database
bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']

# print(len(valid_Inception))
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

mean = 0
for x in range (0,10):
    #Create the model
    InceptionV3_model = Sequential()
    #Add global average pooling layer to reduce the number of input features to the model
    InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
    #Final softmax model for predictions
    InceptionV3_model.add(Dense(133, activation='softmax'))

    InceptionV3_model.summary()

    ### TODO: Compile the model.
    from keras.callbacks import ModelCheckpoint  

    #Specify the optimizer, the loss function and metrics to monitor performance
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)
    time1 = datetime.now()
    InceptionV3_model.fit(train_InceptionV3, train_targets, 
              validation_data=(valid_InceptionV3, valid_targets),
              epochs=20, batch_size=20, callbacks=[checkpointer], verbose=0)
#     ================
    time2 = datetime.now()
    elapsedTime = time2 - time1
    print(str(x)+". Elapsed time: "+ str(elapsedTime.total_seconds()) + " secs")
    mean = mean + elapsedTime.total_seconds()
mean = mean/10
print("Mean: "+ str(mean) + " secs")


mean = 0
for x in range (0,10):
    ### TODO: Load the model weights with the best validation loss.
    InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
    time1 = datetime.now()
    # get index of predicted dog breed for each image in test set
    InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
#     ================
    time2 = datetime.now()
    elapsedTime = time2 - time1
    print(str(x)+". Elapsed time: "+ str(elapsedTime.total_seconds()) + " secs")
    mean = mean + elapsedTime.total_seconds()
mean = mean/10
print("Mean: "+ str(mean) + " secs")

