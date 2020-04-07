import tensorflow as tf
from mnist import create_model
from hyperparameters import learning_rate, batch_size
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide checkpoint name")
        exit()

    model = create_model(learning_rate)
    model.load_weights(f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')
    # Loading data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Data normalization from [0,255] to [0.0,1.0]
    x_test_normalized = x_test/255.0

    model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)
