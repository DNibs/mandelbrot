# Visualizes the Mandelbrot set
# Author: David Niblick
# 06JUN2021

import numpy as np
from matplotlib import pyplot as plt

# todo: fix axis - mismatch between x/y axis, but as long as img is square it's not issue
XMIN = -1.
XMAX = 1.
YMIN = -1.
YMAX = 1.
ROW = 1000
COL = 1000
NUM_TICKS = 10
ITERATIONS = 500
BAILOUT = 4


def linear_mapping(location, old_low, old_high, new_low, new_high):
    return (location - old_low) / (old_high - old_low) * (new_high - new_low) + new_low


def sample_mandelbrot(img_array, x_low, x_high, y_low, y_high):
    x_axis_pixels, y_axis_pixels = np.shape(img_array)
    for m in range(0, x_axis_pixels):
        print('\rImage {0:2.0f}% complete'.format(100 * m / x_axis_pixels), end='')
        for n in range(0, y_axis_pixels):
            count = 0
            z = 0
            x_val = linear_mapping(m, 0, x_axis_pixels, x_low, x_high)
            y_val = linear_mapping(n, 0, y_axis_pixels, y_low, y_high)
            c = x_val + y_val*1.j
            while count < ITERATIONS and abs(z) < BAILOUT:
                count += 1
                z = z**2 + c
            img_array[m, n] = count


def display_image(img_array):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(np.arange(0, ROW, ROW / NUM_TICKS))
    ax.set_yticks(np.arange(0, COL, COL / NUM_TICKS))
    xlabels = np.arange(XMIN, XMAX, (XMAX - XMIN) / NUM_TICKS)[0:NUM_TICKS]
    ax.set_xticklabels([str(round(float(xlabel), 5)) for xlabel in xlabels], rotation=45, ha='right')
    ylabels = np.arange(YMIN, YMAX, (YMAX - YMIN) / NUM_TICKS)[0:NUM_TICKS]
    ax.set_yticklabels([str(round(float(ylabel), 5)) for ylabel in ylabels])
    im = ax.imshow(img_array.T, cmap="RdBu", interpolation="bilinear")
    plt.show()


if __name__ == '__main__':
    mandelbrot_img = np.zeros([COL, ROW])
    sample_mandelbrot(mandelbrot_img, XMIN, XMAX, YMIN, YMAX)
    display_image(mandelbrot_img)
