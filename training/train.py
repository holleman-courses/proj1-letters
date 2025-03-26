import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data_dir = "dataset"
print("Classes found:", os.listdir(data_dir))

for category in os.listdir(data_dir):
    path = os.path.join(data_dir, category)
    print(f"{category} contains {len(os.listdir(path))} images.")

#Image size for resizing
IMG_SIZE = (32, 32)
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

#Load data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

#CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),  # Replaces Flatten() for lower parameter count

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Regularization to prevent overfitting
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks: Early stopping & checkpointing
#checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=10,
    validation_data=val_data,
    validation_steps=len(val_data),
    #callbacks=[checkpoint, early_stopping]
)

model.save('trained_model.h5')
print("Model saved as 'trained_model.h5'.")
model.summary()

#Save tflite model
import numpy as np
import tensorflow as tf 
import tensorflow.lite as tflite

# Load trained model
model = tf.keras.models.load_model("trained_model.h5")

# Number of calibration steps
num_calibration_steps = 100  

# Convert model to TFLite with full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset function (for images)
IMG_SIZE = (32, 32)  # Match input size of CNN
def representative_dataset_gen():
    for _ in range(num_calibration_steps):
        data = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)  # Simulated image
        yield [data]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 

# Ensure 8-bit integer inputs/outputs
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8  

# Convert and save model
tflite_quant_model = converter.convert()
tflite_model_filename = "trained_model.tflite"

with open(tflite_model_filename, "wb") as fpo:
    fpo.write(tflite_quant_model)

print(f"Saved quantized TFLite model as {tflite_model_filename}")