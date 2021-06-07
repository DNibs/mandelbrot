# Visualizes the Mandelbrot set
# Author: David Niblick
# 06JUN2021

import cmath
import numpy as np
from matplotlib import pyplot as plt


PIXEL_LENGTH = 500
TOP_LEFT_PT = (-1, 0)
M_SCALE = 0.3


class Mandelbrot():
    def __init__(self, pixel_length, top_left=(-1, 1), scale=1, bailout=4, max_iterations=1000):
        self.pixel_length = pixel_length
        self.img = np.zeros([self.pixel_length, self.pixel_length])
        self.top_left = top_left
        self.scale = scale
        self.bailout = bailout
        self.max_iterations = max_iterations

    def map_pixel(self, pixel):
        x = self.top_left[0] + (pixel[0] * self.scale)/self.pixel_length
        y = self.top_left[1] - (pixel[1] * self.scale)/self.pixel_length
        return complex(x, y)

    def sample_image(self):
        for m in range(0, self.pixel_length):
            print('\rImage {0:2.0f}% complete'.format(100*m/self.pixel_length), end='')
            for n in range(0, self.pixel_length):
                count = 0
                c = self.map_pixel((m, n))
                z = 0
                while abs(z) < self.bailout and count < self.max_iterations:

                    z = z**2 + c
                    count += 1
                self.img[n, m] = count / self.max_iterations

    def diplay_img(self):
        fig1 = plt.figure()
        plt.imshow(self.img)
        plt.show()

    def sample_and_display(self):
        self.sample_image()
        self.diplay_img()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mandelbrot1 = Mandelbrot(pixel_length=PIXEL_LENGTH, top_left=TOP_LEFT_PT, scale=M_SCALE)
    mandelbrot1.sample_and_display()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
