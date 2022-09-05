#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import binary_accuracy

## Model location
model_path = '../checkpoints/cat_dog_classifier20_0.879.h5'

## Set dataset path
train_path = '../dataset/train/'
val_path = '../dataset/validation/'

## Image size
image_shape = 227

## Channels
channel = 3

## Set image data generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)


## Train dataset
train_ds = train_gen.flow_from_directory(
    train_path, 
    target_size=(image_shape, image_shape),
    batch_size=4,
    shuffle=True,
    class_mode='binary')
    

## Validation dataset
val_ds = val_gen.flow_from_directory(
    val_path,
    target_size=(image_shape, image_shape),
    batch_size=4,
    shuffle=True,
    class_mode='binary')


## Load model trained in Keras (Tensorflow 2)
float_model=tf.keras.models.load_model(model_path)

## Model summary
float_model.summary()

## Quantize the model
quantizer = vitis_quantize.VitisQuantizer(float_model)


quantized_model = quantizer.quantize_model(calib_dataset=train_ds, 
                                           calib_batch_size=4, 
                                           replace_sigmoid=True,
                                           input_shape="?,227,227,3",
                                           weight_bit=8,
                                           bias_bit=8)


quantized_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics= ['binary_accuracy'])


## Evaluate quantized model
quantized_model.evaluate(val_ds)


## Save quantized model
quantized_model.save('quantized_model.h5')
print('Quantized model saved')

## Show model summary
quantized_model.summary()




