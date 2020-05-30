#!/usr/bin/env python3

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Conv2D 
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Activation           
from keras.optimizers import Adam
(X_train , y_train ) , (X_test , y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_train.shape)
print(y_test.shape)

img_rows = X_train[0].shape[0]
img_colms = X_train[1].shape[0]

X_train = X_train.reshape(X_train.shape[0] , img_rows , img_colms , 1 )
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0] , img_rows , img_colms , 1 )
print(X_test.shape)
input_shape = (img_rows , img_colms , 1)
print(input_shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


model= Sequential()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes=y_test.shape[1]
num_pix = y_train.shape[1]*X_train.shape[2]

model.add(Conv2D(50 , (9 , 9 ),
                padding = 'same',
                 input_shape=input_shape))

model.add(Conv2D(25 , ( 7 , 7 ),
                padding = 'same'))


model.add(Conv2D(15 , ( 5 , 5 ),
                padding = 'same'))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(3,3)))
print(model.summary())

model.add(Flatten())
model.add(Dense(units = 512, activation= 'relu' ))

model.add(Dense(units = 256, activation= 'relu' ))
model.add(Dense(units = 128, activation= 'relu' ))
model.add(Dense(units = 64, activation= 'relu' ))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())
           
model.compile(loss = 'categorical_crossentropy',
             optimizer =Adam(),
              metrics=['accuracy'] )

history =  model.fit(X_train , y_train , epochs=5 , validation_data=(X_test , y_test))

import numpy as np

acc = np.array(history.history['val_accuracy'])
acc = max(acc)
z = str(acc)
print("accuracy = " + z, file = open('output.txt' , 'a'))

