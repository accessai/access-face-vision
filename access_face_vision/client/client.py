from PIL import Image
import numpy as np


class AFV(object):

    def encode_image(self, img_path):
        img = Image.open(img_path)
        img_arr = np.array(img)
