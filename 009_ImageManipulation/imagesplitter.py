import os
import numpy as np
from PIL import Image

class ImageSplitter():
    it = 0

    def __init__(self):
        self._image_width = 1600
        self._image_height = 1200
        self._new_image_size = 90
        self._pocket = 10

    def split_into_images(self, image_path):
        self._make_sure_directory_exists()
        img = np.array(Image.open(f'Img/{image_path}'))
        number_of_moves_horizontally = int(self._image_width/(self._new_image_size-self._pocket))
        number_of_moves_vertically = int(self._image_height/(self._new_image_size-self._pocket))
        for j in range(number_of_moves_vertically):
            for i in range(number_of_moves_horizontally):
                horizontal_start = i*(self._new_image_size - self._pocket)
                vertical_start = j*(self._new_image_size - self._pocket)
                tmp_img = Image.fromarray(np.copy(img[vertical_start:vertical_start+self._new_image_size,horizontal_start:horizontal_start+self._new_image_size]))
                tmp_img.save(f"split/{ImageSplitter.it}_{'0' if j < 10 else ''}{j}_{'0' if i < 10 else ''}{i}_tmp.jpg")
        ImageSplitter.it+=1

    def _make_sure_directory_exists(self):
        if not os.path.isdir("Img"):
            raise Exception('Images directory does not exist')
        if not os.path.isdir("split"):
            os.mkdir("split")
