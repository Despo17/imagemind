# visionnet_transfer.py (Memory-optimized using ImageDataGenerator streaming)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create ImageDataGenerators with resizing and preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Resize images on-the-fly to 96x96 (avoid full memory load)
train_gen = train_datagen.flow(
    np.array([tf.image.resize(img, [96, 96]).numpy() for img in x_train]),
    y_train,
    batch_size=64
)

test_gen = test_datagen.flow(
    np.array([tf.image.resize(img, [96, 96]).numpy() for img in x_test]),
    y_test,
    batch_size=64,
    shuffle=False
)

# MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

# Custom classifier
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen
)

# Evaluate
loss, acc = model.evaluate(test_gen, verbose=2)
print(f"\n✅ Test Accuracy with Transfer Learning: {acc:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictions
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = y_test.flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the model
model.save("visionnet_tl.h5")
print("\n✅ Model saved as visionnet_tl.h5")
