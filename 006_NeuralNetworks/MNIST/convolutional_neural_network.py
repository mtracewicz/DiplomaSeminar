from hyperparameters import learning_rate, epochs, batch_size, validation_split
import sys
import tensorflow as tf
import tensorflowjs as tfjs


def create_model(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=15, kernel_size=2,
                                     padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                     padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2,
                                     padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


def save_model(model):
    # Saving compiled model
    tf.keras.models.save_model(
        model, f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')
    # Also for JS
    tfjs.converters.save_keras_model(model, f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}_JS/')

    # and its topograpy
    with open(f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}.json', "w") as file:
        file.write(model.to_json())
        file.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide checkpoint name")
        exit()

    # Loading data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Data normalization from [0,255] to [0.0,1.0]
    # and reshaping it for proper dimensions
    x_train_normalized = (x_train/255.0).reshape(x_train.shape[0], 28, 28, 1)
    x_test_normalized = (x_test/255.0).reshape(x_test.shape[0], 28, 28, 1)
    # Establish the model's topography.
    model = create_model(learning_rate)
    # Train the model on the normalized training set.
    model.fit(x=x_train_normalized, y=y_train, batch_size=batch_size,
              epochs=epochs, shuffle=True,
              validation_split=validation_split)
    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
    save_model(model)
