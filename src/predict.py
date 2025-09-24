import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

(_, _), (x_test, y_test) = datasets.cifar10.load_data()
x_test = x_test / 255.0

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

model = tf.keras.models.load_model("../models/cnn_model.h5")

prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prob_model.predict(x_test)

plt.imshow(x_test[0])
plt.title("Prediction: " + class_names[np.argmax(predictions[0])])
plt.show()
