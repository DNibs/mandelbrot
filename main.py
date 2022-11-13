# Visualizes the Mandelbrot set
# Author: David Niblick
# 06JUN2021

import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

# center_point = (-.1, .93)
center_point = (-.7499, .0314)
row_pixels = 1600
column_pixels = 3840
scale_pixels = .000001  # this is distance betweeen each pixel
# XMIN, XMAX, YMIN, YMAX = (-.12, -.08, .92, .96)
# ROW, COL = (3840, 1600)
NUM_TICKS = 10
ITERATIONS = 250
BAILOUT = 4


def define_box(scale, center_x, center_y, num_rows, num_columns):
    x_left = center_x - (scale * num_columns / 2)
    x_right = center_x + (scale * num_columns / 2)
    y_down = center_y - (scale * num_rows / 2)
    y_up = center_y + (scale * num_rows / 2)
    return x_left, x_right, y_down, y_up


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
    # ax.set_xticks(np.arange(0, num_cols, num_cols / num_ticks))
    # ax.set_yticks(np.arange(0, num_rows, num_rows / num_ticks))
    # xlabels = np.arange(x_min, x_max, (x_max - x_min) / num_ticks)[0:num_ticks]
    # ax.set_xticklabels([str(round(float(xlabel), 5)) for xlabel in xlabels], rotation=45, ha='right')
    # ylabels = np.arange(y_min, y_max, (y_max - y_min) / num_ticks)[0:num_ticks]
    # ax.set_yticklabels([str(round(float(ylabel), 5)) for ylabel in ylabels])
    ax.set_axis_off()
    im = ax.imshow(np.log(img_array), cmap="gray", interpolation="bilinear")
    plt.show()


def img_post_process(img_array):
    hist, discard = np.histogram(img_array, bins=range(256))
    # check and invert to make img dark
    low_color_count = np.sum(hist[:128])
    high_color_count = np.sum(hist[129:])
    if high_color_count > low_color_count:
        img_array = 256 - img_array

    # Drop floor to darkest color
    img_array = img_array - np.min(img_array)
    return img_array.astype(int)


if __name__ == '__main__':
    mandelbrot_img = np.zeros([row_pixels, column_pixels])
    box = define_box(scale_pixels, center_point[0], center_point[1], row_pixels, column_pixels)
    print('box limits (x limits, y limits): ', box)
    sample_mandelbrot(mandelbrot_img, box[0], box[1], box[2], box[3])
    new_mandelbrot_img = img_post_process(mandelbrot_img)
    display_image(new_mandelbrot_img, box[0], box[1], box[2], box[3], num_ticks=NUM_TICKS)
    timestamp = time.time()
    print('img number: ', timestamp)
    cv2.imwrite('cv_output_{}.png'.format(timestamp), new_mandelbrot_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
