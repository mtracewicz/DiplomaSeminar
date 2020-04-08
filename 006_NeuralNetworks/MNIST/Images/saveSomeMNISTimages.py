from keras.preprocessing.image import save_img
import tensorflow as tf
import random
import time
from sys import argv
from progress.bar import ChargingBar

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
data = x_train.reshape(x_train.shape[0], 28, 28, 1)

number_of_images = int(argv[1]) if len(argv) == 2 else 10

random.seed(time.time)
with ChargingBar("Downloading", max=number_of_images) as bar:
    for i in range(number_of_images):
        save_img(f'./test_images/test{i}.jpg', data[random.randint(0, 10000)])
        bar.next()
