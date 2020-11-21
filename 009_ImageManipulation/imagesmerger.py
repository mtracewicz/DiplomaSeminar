import os
import numpy as np
from PIL import Image

class ImagesMerger():
    def __init__(self, src_directory, out_directory):
        self._image_names = os.listdir(src_directory)
        self._image_names.sort()
        tmp = list(dict.fromkeys([img[0] for img in self._image_names]))
        self.number_of_images_to_create = len(tmp)
        self._out_directory = out_directory

    def merge():
        for i,val in enumerate(self._image_names):
            
