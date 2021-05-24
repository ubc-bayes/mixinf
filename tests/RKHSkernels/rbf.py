import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad

# banwidth of kernel
k_gamma = 0.5
#k_gamma = 700
# define rbf kernel
def kernel(x, y):
    return np.exp(-0.5 * np.sum((x - y)**2, axis = -1) / k_gamma) # returns (N,1)

# derivatives of the kernel
def dk_x(x, y): return egrad(lambda t : kernel(t, y))(x) # returns (N,K)
def dk_y(x, y): return egrad(lambda t : kernel(x, t))(y) # returns (N,K)
def dk_xy(x, y):
    # returns (N,1)
    g = 0
    for d in range(x.shape[1]):
        g += egrad(lambda t : dk_x(x,t)[:, d])(y)[:, d]
    return g
