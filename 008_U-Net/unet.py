from hyperparameters import learning_rate, epochs, batch_size, validation_split
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def create_model(input_img,n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_model(model):
    # Saving compiled model
    tf.keras.models.save_model(
        model,
        f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}'
    )
    # and its topography
    with open(f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}.json', "w") as file:
        file.write(model.to_json())
        file.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide checkpoint name")
        exit()

    IMAGES_DIRECTORY= os.getcwd()+'/008_U-Net/Img'
    filenames = os.listdir(IMAGES_DIRECTORY)
    number_of_images = len(filenames)
    x_train = np.ones((number_of_images,1200,1600,3))
    for i,filename in enumerate(filenames):
        img = Image.open(f"{IMAGES_DIRECTORY}/{filename}")
        x_train[i] = np.array(img)

    y_train = np.copy((255.0 - x_train)/255.0).reshape(x_train.shape[0], 1200, 1600, 3)
    # Data normalization from [0,255] to [0.0,1.0]
    # and reshaping it for proper dimensions
    x_train_normalized = (x_train/255.0).reshape(x_train.shape[0], 1200, 1600, 3)
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
    res = Image.fromarray(((model.predict(x_train_normalized[0]))[0] * 255).astype(np.uint8))
    res.save("res.jpeg")

    # Evaluate against the test set.
    # print("\n Evaluate the new model against the test set:")
    # model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
    # save_model(model)
