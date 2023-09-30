# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:48:37 2023

@author: Asus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

from tp1_utils import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, MobileNet, VGG16, VGG19, MobileNetV2, Xception,ResNet152V2, NASNetMobile

def training_plot(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy ')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss ')
  plt.legend()
  plt.show()
  
def training_plot_labels(history):
  acc = history.history['binary_accuracy']
  val_acc = history.history['val_binary_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy ')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss ')
  plt.legend()
  plt.show()


data = load_data()

train_X = data["train_X"]
test_X = data["test_X"]
train_masks = data["train_masks"]
test_masks = data["test_masks"]
train_classes = data["train_classes"]
train_labels = data["train_labels"]
test_classes = data["test_classes"]
test_labels = data["test_labels"]

result = []

preds = []


NUM_EPOCHS = 100


def CNN_V1(act):
    model = models.Sequential()
    model.add(layers.Normalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(BatchNormalization())
    
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))

    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(10, activation=act,
              kernel_initializer=tf.keras.initializers.GlorotNormal()))

    return model

def CNN_V2(act):
    model = models.Sequential()
    model.add(layers.Normalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dropout(0.3))
    
    model.add(BatchNormalization())
    
    model.add(layers.Dense(512, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(64, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(32, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(10, activation=act,
              kernel_initializer=tf.keras.initializers.GlorotNormal()))

    return model

def CNN_V3(act):
    model = models.Sequential()
    model.add(layers.Normalization(input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same",
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    
    model.add(layers.Dropout(0.3))
    
    model.add(BatchNormalization())
    
    model.add(layers.Dense(512, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(128, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))

    model.add(layers.Dense(64, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(32, activation='relu',
              kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(layers.Dense(10, activation=act,
              kernel_initializer=tf.keras.initializers.GlorotNormal()))

    return model

def mlp():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.Dense(10, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotNormal())
    ])
    
    model.compile(Adam(),loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_X, train_classes, epochs=50, validation_split=0.125)

    r = model.evaluate(test_X, test_classes, verbose=1)
    
    pred = model.predict(test_X)
    
    preds.append(pred)
    
    training_plot(history)

def run_classes_train():
    
    print("START")
    print("Tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0001)
    Early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
    
    model = CNN_V1("softmax")

    model.compile(Adam(learning_rate=0.001),loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_X, train_classes, epochs=NUM_EPOCHS, validation_split=0.125, batch_size=32, callbacks=[reduce_lr, Early])

    r = model.evaluate(test_X, test_classes, verbose=1)
    
    pred = model.predict(test_X)
    
    preds.append(pred)
    
    training_plot(history)
    
def run_classes_best_param():
    
    model = CNN_V1("softmax")
    
    model.compile(Adam(learning_rate=0.001),loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.load_weights('classesWeights.h5')
    
    r = model.evaluate(test_X, test_classes, verbose=1)
    
    pred = model.predict(test_X)
    
    preds.append(pred)
    
def run_labels_train():
    
    print("START")
    print("Tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0001)
    Early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
    
    model = CNN_V1("sigmoid")

    model.compile(Adam(learning_rate=0.001),loss="binary_crossentropy", metrics=["binary_accuracy"])
    history = model.fit(
        train_X, train_labels, epochs=NUM_EPOCHS, validation_split=0.125, batch_size=32, callbacks=[reduce_lr, Early])

    r = model.evaluate(test_X, test_labels, verbose=1)
    
    pred = model.predict(test_X)
    
    preds.append(pred)
    
    training_plot_labels(history)
    
def run_labels_best_param():
    
    model = CNN_V1("sigmoid")
    
    model.compile(Adam(learning_rate=0.001),loss="binary_crossentropy", metrics=["binary_accuracy"])

    model.load_weights('labelsWeights.h5')
    
    r = model.evaluate(test_X, test_labels, verbose=1)
    
    pred = model.predict(test_X)
    
    preds.append(pred)
    
def ImageNet():
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Extract features from the last convolutional layer
    x = base_model.output

    # Flatten the output
    x = layers.Flatten()(x)

    # Add a fully connected layer with 128 units
    x = layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(x)

    # Apply batch normalization
    x = layers.BatchNormalization()(x)

    # Apply dropout regularization
    x = layers.Dropout(0.5)(x)

    # Add another fully connected layer with 32 units
    x = layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(x)

    # Apply batch normalization
    x = layers.BatchNormalization()(x)

    # Add a final fully connected layer with 10 units
    output = layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal())(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    return model


def run_preTrained():

    op = Adam(beta_1=0.95)

    model = ImageNet()

    model.compile(optimizer=op,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(
              train_X, train_classes, epochs=150, validation_split=0.125)

    model.evaluate(test_X, test_classes, verbose=1)
    
    training_plot(history)