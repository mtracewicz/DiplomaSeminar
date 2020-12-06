import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from hyperparameters import batch_size, epochs, learning_rate, validation_split
from model import get_model, save_model

def load_images(dir: str) -> np.array:
    IMAGES_DIRECTORY= os.path.join(os.getcwd(), dir)
    filenames = os.listdir(IMAGES_DIRECTORY)
    number_of_images = len(filenames)
    images = np.ones((number_of_images,200,200, 3))
    for i, filename in enumerate(filenames):
        img = Image.open(f"{IMAGES_DIRECTORY}/{filename}")
        images[i] = np.array(img)[:,:,:3]
    return images


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unet.py inputs_directory truths_directory checkpoint_name")
        exit()

    #Loading data
    x_train = load_images(sys.argv[1])
    y_train = load_images(sys.argv[2])

    # Data normalization from [0,255] to [0.0,1.0]
    # and reshaping it for proper dimensions
    x_train= (x_train/255.0).reshape(x_train.shape[0], 200, 200, 3)
    y_train= (y_train/255.0).reshape(y_train.shape[0], 200, 200, 3)

    # Seting tf floating point operations to 64 use bits
    tf.keras.backend.set_floatx('float64')

    # Establish the model's topography
    model = get_model(x_train[0].shape)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

    # Save model
    save_model(model, sys.argv[3])
    print('Finished!')
