import tensorflow as tf
import numpy as np
from PIL import Image

import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usagepredict.py model file")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')
    # Loading data
    image = Image.open(sys.argv[2])
    data = np.array(image, dtype =np.float64)
    data.reshape((1,200,200,3))
    prediction = model.predict(data)
    res = Image.fromarray((prediction * 255).astype(np.uint8))
    res.save("res.jpeg")
