# Visualizes the Mandelbrot set
# Author: David Niblick
# 06JUN2021

import cmath
import numpy as np
from matplotlib import pyplot as plt


PIXEL_LENGTH = 500


class Mandelbrot():
    def __init__(self, pixel_length, top_left=(1, 1), scale=1):
        self.pixel_length = pixel_length
        self.img = np.zeros([self.pixel_length, self.pixel_length])
        self.top_left = top_left
        self.scale = scale



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mandelbrot1 = Mandelbrot(pixel_length=PIXEL_LENGTH)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
