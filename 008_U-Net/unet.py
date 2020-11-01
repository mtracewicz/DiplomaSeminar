import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from hyperparameters import batch_size, epochs, learning_rate, validation_split
from model import create_model,save_model

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
    y_train = np.copy((255.0 - x_train)/255.0).reshape(x_train.shape[0], 1200, 1600, 3)

    # Establish the model's topography.
    model = create_model(x_train)
    # Train the model on the normalized training set.
    model.fit(
        x=x_train_normalized,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True
    )
    # Save model
    save_model(model,sys.argv[1])

    print('Finished!')
