import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set up paths
data_path = "Dataset/Original Images"
classes = ["Others", "Monkey Pox"]

# Read data into arrays
data = []
labels = []
for class_name in classes:
    class_path = os.path.join(data_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        data.append(img)
        if class_name == "Others":
            labels.append(0)
        else:
            labels.append(1)
data = np.array(data)
labels = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Preprocess data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

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
    epochs=50,
    validation_data=valid_data,
    callbacks=cb
)

model.save('monkeypox_model.h5')