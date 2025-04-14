import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Define input shape and number of classes
input_shape = (28, 28, 3)
num_classes = 44

# Define DenseNet-8 model
def DenseNet8(num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense block 1
    x = dense_block(x, 64, 2)

    # Transition layer 1
    x = transition_layer(x, 128)

    # Dense block 2
    x = dense_block(x, 128, 2)

    # Transition layer 3
    x = transition_layer(x, 256)

    # Dense block 4
    x = dense_block(x, 256, 2)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Dense block function
def dense_block(x, num_filters, num_layers):
    for i in range(num_layers):
        conv = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
        x = layers.Concatenate()([x, conv])
    return x

# Transition layer function
def transition_layer(x, num_filters):
    x = layers.Conv2D(num_filters, 1, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    return x

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/final_train',
    target_size=(28, 28),
    batch_size=10,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    'data/final_validation',
    target_size=(28, 28),
    batch_size=10,
    class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
    'data/final_test',
    target_size=(28, 28),
    batch_size=10,
    class_mode='categorical')

# Train model
model = DenseNet8(num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_generator, epochs=60, validation_data=validation_generator)
# Retrieve accuracy and validation accuracy values from history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot the accuracy and validation accuracy
epochs = range(1, len(accuracy) + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary of the model
model.summary()


