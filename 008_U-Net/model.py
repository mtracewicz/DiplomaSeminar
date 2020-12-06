import tensorflow as tf


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",)(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def get_model(input_size):
    n_filters=2
    dropout=0.5
    batchnorm=True

    inputs = tf.keras.Input(input_size)
    # contracting path
    c1 = conv2d_block(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u5 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c4)
    u5 = tf.keras.layers.concatenate([u5, c3])
    u5 = tf.keras.layers.Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u6 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = tf.keras.layers.concatenate([u7, c1], axis=3)
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c7)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def save_model(model, checkpoint_name, to_json=True):
    # Saving compiled model
    tf.keras.models.save_model(
        model,
        f'./checkpoints/{checkpoint_name}/{checkpoint_name}'
    )
    # and its topography
    if to_json:
        with open(f'./checkpoints/{checkpoint_name}/{checkpoint_name}.json', "w") as file:
            file.write(model.to_json())
            file.flush()
