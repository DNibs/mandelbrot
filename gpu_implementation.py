from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math

#XMIN, XMAX, YMIN, YMAX = (-.378, -.369, -.664, -.655)
XMIN, XMAX, YMIN, YMAX = (-1.6, 0.6, -1, 1)
XMIN, XMAX, YMIN, YMAX = (-.479, -.476, .0493, .0523)
# XMIN, XMAX, YMIN, YMAX = (-.236, -.23, -.762, -.756)


# center_point = (-.1, .93)
center_pixel_x = -.7499
center_pixel_y = .0314
row_pixels = 1600
column_pixels = 3840
scale_pixels = .000001  # this is distance betweeen each pixel
# XMIN, XMAX, YMIN, YMAX = (-.12, -.08, .92, .96)
# ROW, COL = (3840, 1600)
NUM_TICKS = 10
ITERATIONS = 250
BAILOUT = 4


def define_bounding_box(scale, center_x, center_y, num_rows, num_columns):
    """
    Returns array for the bounding box based on center point, size of image, and scale of each pixel
    """
    x_left = center_x - (scale * num_columns / 2)
    x_right = center_x + (scale * num_columns / 2)
    y_down = center_y - (scale * num_rows / 2)
    y_up = center_y + (scale * num_rows / 2)
    return x_left, x_right, y_down, y_up


def determine_kernal_box(num_rows, num_columns, thread_x=32, thread_y=32):
    """
    Returns block size (x, y) and thread size (x, y) with lazy round up of one
    """
    return ((num_columns//thread_x)+1, (num_rows//thread_y)+1), (thread_x, thread_y)


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


def display_image(img_array, x_min, x_max, y_min, y_max, num_ticks=10):
    """
    Displays the image but does not save. uncomment the axis to get location on mandelbrot space for better
    bounding box selection
    """
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


@cuda.jit
def mandelbrot_kernel(img, x_low, x_high, y_low, y_high):
    """
    img is img array, bound box is mandelbrot coordinates from define_bounding_box
    x dim is associated with number of columns (img[1], y dim is associated with number of rows (img[0])
    """
    pixel_x, pixel_y = cuda.grid(2)

    if pixel_y >= img.shape[0] or pixel_x >= img.shape[1]:
        return

    mand_loc_x = pixel_x / img.shape[1] * (x_high - x_low) + x_low
    mand_loc_y = pixel_y / img.shape[0] * (y_high - y_low) + y_low

    c = np.complex(mand_loc_x, mand_loc_y)
    z = np.complex(0., 0.)
    for i in range(ITERATIONS):
        z = (z ** 2) + c
        if math.sqrt(z.real ** 2 + z.imag ** 2) >= BAILOUT:
            img[pixel_y, pixel_x] = i
            break
        else:
            img[pixel_y, pixel_x] = ITERATIONS


mandelbrot_img = np.zeros([row_pixels, column_pixels])
kernel_dim = determine_kernal_box(row_pixels, column_pixels)
box = define_bounding_box(scale_pixels, center_pixel_x, center_pixel_y, row_pixels, column_pixels)
print(kernel_dim)
print(box)
print(np.max(mandelbrot_img))
print(np.min(mandelbrot_img))
mandelbrot_kernel[kernel_dim](mandelbrot_img, box[0], box[1], box[2], box[3])
new_img = img_post_process(mandelbrot_img)
print(np.max(new_img))
print(np.min(new_img))
timestamp = time.time()
print('img number: ', timestamp)
cv2.imwrite('cv_output_{}.png'.format(timestamp), new_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
