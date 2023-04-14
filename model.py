import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2

# Data Generators
train_gen = ImageDataGenerator(rescale=1./255)
valid_gen = ImageDataGenerator(rescale=1./255,validation_split=0.4)

# Loading Data
train_data = train_gen.flow_from_directory("./Dataset/Augmented Images/Augmented Images",target_size=(256,256),shuffle=True,class_mode='binary')
valid_data = valid_gen.flow_from_directory('./Dataset/Original Images',target_size=(256,256),shuffle=True,subset='training',class_mode='binary')
test_data = valid_gen.flow_from_directory('./Dataset/Original Images',target_size=(256,256),shuffle=True,subset='validation',class_mode='binary')
# Define model architecture
base_model = ResNet50V2(
    include_top=False,
    input_shape=(256,256,3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256,activation='relu'),
    BatchNormalization(),
    Dense(164,activation='relu'),
    BatchNormalization(),
    Dense(1,activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

cb = [EarlyStopping(patience=5,monitor='val_accuracy',mode='max',restore_best_weights=True),ModelCheckpoint("ResNet50V2-01.h5",save_best_only=True)]

model.fit(
    train_data,
    epochs=5,
    validation_data=valid_data,
    callbacks=cb
)

model.save('monkeypox_model.h5')