'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

#%%

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 4281
nb_validation_samples = 468
epochs = 50
batch_size = 16


#%%
datagen = ImageDataGenerator(rescale=1. / 255)

print('llego')
# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_train.npy', 'w'),
        bottleneck_features_train)

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // batch_size)
np.save(open('bottleneck_features_validation.npy', 'w'),
        bottleneck_features_validation)

print('llego')

#%%

train_data = np.load(open('bottleneck_features_train.npy'))
train_labels = np.array(
    [0] * (237) + [1] * (351) + [2] * (259) + [3] * (550) + [4] * (199) + [5] * (428) + [6] * (589) + [7] * (199) + [8] * (465) + [9] * (208) + [10] * (449) + [11] * (347))

#validation_data = np.load(open('bottleneck_features_validation.npy'))
#validation_labels = np.array(
#    [0] * (26) + [1] * (39) + [2] * (28) + [3] * (60) + [4] * (22) + [5] * (47) + [6] * (65) + [7] * (22) + [8] * (51) + [9] * (23) + [10] * (47) + [11] * (38))
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

#%%

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1)

#model.fit_generator(
#    train_data,
#    steps_per_epoch=nb_train_samples // batch_size,
#    epochs=epochs,
#    validation_data=(validation_data, validation_labels),
#    validation_steps=nb_validation_samples // batch_size)


model.save_weights(top_model_weights_path)


