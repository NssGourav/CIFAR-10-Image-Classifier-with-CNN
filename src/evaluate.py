import tensorflow as tf
from tensorflow.keras import datasets

(_, _), (x_test, y_test) = datasets.cifar10.load_data()
x_test = x_test / 255.0

model = tf.keras.models.load_model("../models/cnn_model.h5")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test Accuracy:", test_acc)
