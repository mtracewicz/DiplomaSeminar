import tensorflow as tf


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = tf.keras.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",)(input_tensor)
    x = tf.keras.Activation("relu")(x)
    # second layer
    x = tf.keras.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    x = tf.keras.Activation("relu")(x)
    return x


def create_model(input_img,n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.MaxPooling2D((2, 2)) (c1)
    p1 = tf.keras.Dropout(tf.keras.dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.MaxPooling2D((2, 2)) (c2)
    p2 = tf.keras.Dropout(tf.keras.dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.MaxPooling2D((2, 2)) (c3)
    p3 = tf.keras.Dropout(tf.keras.dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = tf.keras.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = tf.keras.Dropout(tf.keras.dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = tf.keras.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = tf.keras.concatenate([u6, c4])
    u6 = tf.keras.Dropout(tf.keras.dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = tf.keras.concatenate([u7, c3])
    u7 = tf.keras.Dropout(tf.keras.dropout)(u7)

    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    u8 = tf.keras.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = tf.keras.concatenate([u8, c2])
    u8 = tf.keras.Dropout(tf.keras.dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = tf.keras.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = tf.keras.concatenate([u9, c1], axis=3)
    u9 = tf.keras.Dropout(tf.keras.dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = tf.keras.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = tf.keras.Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer=tf.keras.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model

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