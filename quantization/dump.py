import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = 'quantized_model.h5'

image_shape = 227

dump_gen = ImageDataGenerator(rescale=1./255)

dump_ds = dump_gen.flow_from_directory(
    '../dataset/dump/',
    target_size=(image_shape, image_shape),
    batch_size=4,
    shuffle=False,
    class_mode='binary')


quantized_model = tf.keras.models.load_model(model_path)


vitis_quantize.VitisQuantizer.dump_model(model=quantized_model,
                                         dataset=dump_ds,
                                         dump_float=True,
                                         output_dir='./dump_model')

