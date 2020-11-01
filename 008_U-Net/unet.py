from hyperparameters import learning_rate, epochs, batch_size, validation_split
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image

def create_model(learning_rate):
    # Model creation
    model = tf.keras.models.Sequential()
    # Downsampling
    model.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding='same',
        activation='relu',
        use_bias=True,
        input_shape=(1200, 1600, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    #Flat
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))

    # Upsampling
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    # Flatenning
    model.add(tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        use_bias=True,
        padding='same',
        activation='relu'
    ))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
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
    model = create_model(learning_rate)
    # Train the model on the normalized training set.
    model.fit(
        x=x_train_normalized,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=validation_split
    )
    res = Image.fromarray(((model.predict(x_train_normalized[0]))[0] * 255).astype(np.uint8))
    res.save("res.jpeg")

    # Evaluate against the test set.
    # print("\n Evaluate the new model against the test set:")
    # model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
    # save_model(model)
