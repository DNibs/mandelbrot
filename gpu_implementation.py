from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

#XMIN, XMAX, YMIN, YMAX = (-.378, -.369, -.664, -.655)
XMIN, XMAX, YMIN, YMAX = (-1.6, 0.6, -1, 1)
XMIN, XMAX, YMIN, YMAX = (-.479, -.476, .0493, .0523)
# XMIN, XMAX, YMIN, YMAX = (-.236, -.23, -.762, -.756)
ROW, COL = (1000, 1000)  # cannot have more COL than ROW
NUM_TICKS = 10
ITERATIONS = 200
BAILOUT = 4


@cuda.jit
def mandelbrot_kernel(data, xlow, xhigh, ylow, yhigh):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    def mapFromTo(location, old_low, old_high, new_low, new_high):
        return (location - old_low) / (old_high - old_low) * (new_high - new_low) + new_low

    x = mapFromTo(tx, 0, COL, xlow, xhigh)
    y = mapFromTo(ty, 0, ROW, ylow, yhigh)
    c = complex(x, y)
    z = 0.0j
    for i in range(ITERATIONS):
        z = z ** 3 + c
        if (z.real ** 2 + z.imag ** 2) >= BAILOUT:
            data[ty, tx] = i
            break
        else:
            data[ty, tx] = ITERATIONS


plot = np.zeros([ROW, COL])
mandelbrot_kernel[ROW, COL](plot, XMIN, XMAX, YMIN, YMAX)
fig = plt.figure(dpi=200)
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, COL, COL/NUM_TICKS))
ax.set_yticks(np.arange(0, ROW, ROW/NUM_TICKS))
xlabels = np.arange(XMIN, XMAX, (XMAX-XMIN)/NUM_TICKS)[0:NUM_TICKS]
ax.set_xticklabels([str(round(float(xlabel), 5)) for xlabel in xlabels], rotation=45, ha='right')
ylabels = np.arange(YMIN, YMAX, (YMAX-YMIN)/NUM_TICKS)[0:NUM_TICKS]
ax.set_yticklabels([str(round(float(ylabel), 5)) for ylabel in ylabels])
im = ax.imshow(np.log(plot), cmap="RdBu", interpolation="bilinear")
# im = ax.imshow(plot, cmap="RdBu")
plt.show()
