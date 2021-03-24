import numpy as np

# banwidth of kernel
gamma = 0.1

# define rbf kernel
def kernel(x, y): return np.exp(-(x - y)**2 / gamma)

# derivatives of the kernel
def dk_x(x, y): return 2 * kernel(x, y) * (x - y) / gamma
def dk_y(x, y): return 2 * kernel(x, y) * (y - x) / gamma
def dk_xy(x, y): return 4 * kernel(x, y) * (2*(x - y)**2 + 1) / gamma
