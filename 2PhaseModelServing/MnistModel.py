import pathlib

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import BatchNormalization

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = train_images.reshape(len(train_images), 28, 28, 1).astype('float32')
test_images = test_images.reshape(len(test_images), 28, 28, 1).astype('float32')

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = Sequential()
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2, 1))
model.add(BatchNormalization())

#model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(MaxPooling2D(2, 1))
#model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=1, verbose=1, validation_data=(test_images, test_labels))

model.save("mnist_vgg")

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
print(interpreter.get_tensor_details())
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

tflite_model_dir = pathlib.Path("quantized_model/")
tflite_model_quant_file = tflite_model_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

# Initialize the intepreter
test_images_indices = range(test_images.shape[0])
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
predictions = np.zeros((len(test_images_indices),), dtype=int)

for i, test_image_index in enumerate(test_images_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details['quantization']
        test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predictions[i] = output.argmax()

accuracy = (np.sum(test_labels==predictions) * 100 )/ len(test_images)

print("Quantized model accuracy is %.4f%% (Number of test samples = %d)" % (accuracy, len(test_images)))