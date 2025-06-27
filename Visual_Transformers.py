import keras_cv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ===================== CONFIGURATION =====================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # Carcinoma, Squamous Cell Carcinoma, Adenocarcinoma, Normal
TRAIN_PATH = 'dataset/train'
VAL_PATH = 'dataset/val'
MODEL_PATH = 'vit_liver_model.h5'
LEARNING_RATE = 1e-4

# ===================== DATA LOADING =====================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ===================== BUILD ViT MODEL =====================
vit_model = keras_cv.models.ViTClassifier(
    image_size=IMAGE_SIZE,
    classes=NUM_CLASSES,
    include_rescaling=True,
    activation='softmax'
)

# ===================== COMPILE =====================
vit_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===================== CALLBACKS =====================
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ===================== TRAIN =====================
history = vit_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ===================== SAVE FINAL MODEL =====================
vit_model.save("vit_liver_final_model.h5")
