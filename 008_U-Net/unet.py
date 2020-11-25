import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from hyperparameters import batch_size, epochs, learning_rate, validation_split
from model import create_model2,save_model,myModel

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide checkpoint name")
        exit()

    #Loading data
    IMAGES_DIRECTORY= os.getcwd()+'/008_U-Net/Img'
    filenames = os.listdir(IMAGES_DIRECTORY)
    number_of_images = len(filenames)
    x_train = np.ones((number_of_images,1200,1600,3))
    for i,filename in enumerate(filenames):
        img = Image.open(f"{IMAGES_DIRECTORY}/{filename}")
        x_train[i] = np.array(img)

    # Data normalization from [0,255] to [0.0,1.0]
    # and reshaping it for proper dimensions
    x_train_normalized = (x_train/255.0).reshape(x_train.shape[0], 1200, 1600, 3)
    # Prepare grand truth
    y_train = ((255.0-np.copy(x_train)/255.0).reshape(x_train.shape[0], 1200, 1600, 3))

    tf.keras.backend.set_floatx('float64')
    # Establish the model's topography.
    model = myModel(x_train_normalized)
    model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=["accuracy"])

    # Save model
    save_model(model,sys.argv[1])
    res = Image.fromarray(((model.predict(np.array(Image.open(f'{os.getcwd()}/008_U-Net/Img/przyp. 1 STAT 3 pow 40x.JPG.JPG'))*255))).astype(np.uint8))
    res.save("res.jpeg")
    print('Finished!')
