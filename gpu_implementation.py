from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

XMIN = -.378
XMAX = -.369
YMIN = -.664
YMAX = -.655
ROW = 1000
COL = 1000
NUM_TICKS = 10
ITERATIONS = 100
BAILOUT = 4


@cuda.jit
def mandelbrot_kernel(data, xlow, xhigh, ylow, yhigh):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    def mapFromTo(x, a, b, c, d):
        y = (x - a) / (b - a) * (d - c) + c
        return y

    x = mapFromTo(tx, 0, ROW, xlow, xhigh)
    y = mapFromTo(ty, 0, COL, ylow, yhigh)
    c = complex(x, y)
    z = 0.0j
    for i in range(ITERATIONS):
        z = z ** 2 + c
        if (z.real ** 2 + z.imag ** 2) >= BAILOUT:
            data[tx, ty] = i
            break
        else:
            data[tx, ty] = 100



plot = np.zeros([ROW, COL])
mandelbrot_kernel[ROW, COL](plot, XMIN, XMAX, YMIN, YMAX)
fig = plt.figure(dpi=200)
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(np.arange(0, ROW, ROW/NUM_TICKS))
ax.set_yticks(np.arange(0, COL, COL/NUM_TICKS))
xlabels = np.arange(XMIN, XMAX, (XMAX-XMIN)/NUM_TICKS)[0:NUM_TICKS]
ax.set_xticklabels([str(round(float(xlabel), 5)) for xlabel in xlabels], rotation=45, ha='right')
ylabels = np.arange(YMIN, YMAX, (YMAX-YMIN)/NUM_TICKS)[0:NUM_TICKS]
ax.set_yticklabels([str(round(float(ylabel), 5)) for ylabel in ylabels])
im = ax.imshow(plot.T, cmap="RdBu", interpolation="bilinear")
plt.show()