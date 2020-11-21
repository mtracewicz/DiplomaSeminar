import os
from imagesplitter import ImageSplitter

if __name__ == "__main__":
    splitter = ImageSplitter()
    images = os.listdir("Img")
    images.sort()
    for img in images:
        splitter.split_into_images(img)
