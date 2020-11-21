import os
import numpy as np
from PIL import Image

class ImageSplitter():
    def __init__(self):
        self._image_width = 1600
        self._image_height = 1300
        self._new_image_size = 100
        self._horizontal_pocket = 25
        self._vertical_pocket = 20

    def split_into_images(self, image_path):
        self._make_sure_directory_exists()
        img = np.array(Image.open(image_path))
        number_of_moves_horizontally = self._image_width/(self._new_image_size-self._horizontal_pocket)
        number_of_moves_vertically = self._image_height/(self._new_image_size-self._vertical_pocket)
        for i in range(number_of_moves_horizontally):
            for j in range(number_of_moves_vertically):
                horizontal_start = i*(self._new_image_size - self._horizontal_pocket)
                vertical_start = j*(self._new_image_size - self._vertical_pocket)
                tmp_img = Image.fromarray(np.copy(img[vertical_start+self._new_image_size,horizontal_start+self._new_image_size]))
                tmp_img.save(f"split/{i}_{j}_tmp.jpg")

    def _make_sure_directory_exists(self):
        if not os.direxists("split"):
            os.mkdir("split")