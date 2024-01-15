# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:36:49 2023

@author: ghmèm manar
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
df=pd.read_csv("Dataset.csv")
print(df.head(20))
print(df.columns)
print(df.info)
print(type(df))
print(df.columns)
train_dir="Original Images/Original Images"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=32)
classes = list(train_ds.class_indices.keys())
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes),activation='softmax'))
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ["accuracy"])
model.summary()
#Training
history = model.fit(train_ds,epochs= 30, batch_size=32)
#Prédection
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224, 3))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    
    print("Actual: " + (image_path.split("/")[-1]).split("_")[0])
    print("Predicted: " + classes[np.argmax(pred)])

# Example predictions with correct image paths
predict_image("Original Images/Original Images/Brad Pitt/Brad Pitt_102.jpg")
predict_image("Original Images/Original Images/Henry Cavill/Henry Cavill_28.jpg")

# Métrique d'évaluation
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.xlabel('Time')
plt.legend(['accuracy', 'loss'])
plt.show()
plt.close()



