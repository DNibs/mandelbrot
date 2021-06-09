# Visualizes the Mandelbrot set
# Author: David Niblick
# 06JUN2021

import numpy as np
from matplotlib import pyplot as plt

# todo: fix axis - mismatch between x/y axis, but as long as img is square it's not issue
XMIN, XMAX, YMIN, YMAX = (-.12, -.08, .92, .96)
ROW, COL = (1000, 1000)
NUM_TICKS = 10
ITERATIONS = 100
BAILOUT = 4


def linear_mapping(location, old_low, old_high, new_low, new_high):
    return (location - old_low) / (old_high - old_low) * (new_high - new_low) + new_low


def sample_mandelbrot(img_array, x_low, x_high, y_low, y_high):
    y_axis_pixels, x_axis_pixels = np.shape(img_array)
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
            img_array[n, m] = count


def display_image(img_array, x_min, x_max, y_min, y_max, num_ticks=10):
    num_rows, num_cols = np.shape(img_array)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(np.arange(0, num_cols, num_cols / num_ticks))
    ax.set_yticks(np.arange(0, num_rows, num_rows / num_ticks))
    xlabels = np.arange(x_min, x_max, (x_max - x_min) / num_ticks)[0:num_ticks]
    ax.set_xticklabels([str(round(float(xlabel), 5)) for xlabel in xlabels], rotation=45, ha='right')
    ylabels = np.arange(y_min, y_max, (y_max - y_min) / num_ticks)[0:num_ticks]
    ax.set_yticklabels([str(round(float(ylabel), 5)) for ylabel in ylabels])
    im = ax.imshow(np.log(img_array), cmap="RdBu", interpolation="bilinear")
    plt.show()


if __name__ == '__main__':
    mandelbrot_img = np.zeros([ROW, COL])
    sample_mandelbrot(mandelbrot_img, XMIN, XMAX, YMIN, YMAX)
    display_image(mandelbrot_img, XMIN, XMAX, YMIN, YMAX, num_ticks=NUM_TICKS)
