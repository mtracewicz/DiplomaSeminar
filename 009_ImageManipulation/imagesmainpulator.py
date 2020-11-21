import os
from imagespliter import ImageSplitter

if __name__ == "__main__":
    splitter = ImageSplitter()
    images = os.listdir("Img")
    for img in images:
        splitter.split_into_images(img)
