# train_mask_model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

INIT_LR = 1e-4
EPOCHS = 10
BS = 32
IMG_SIZE = 224

# Load and preprocess data
train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2,
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    "dataset",  # should have "with_mask" and "without_mask" folders
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BS,
    class_mode="binary",
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BS,
    class_mode="binary",
    subset='validation'
)

# Load base model
base = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=x)
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(INIT_LR), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

model.save("mask_detector.model")
