import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad

# these functions build an rbf gaussian kernel such that the bandwidth is equal to the median squared distance of a given sample
# specifically, given a sample, get_gamma gets the median squared distance and get_kernel builds the kernel and its first and second order derivatives

# banwidth of kernel
def get_gamma(x):
    # receives sample of size (N,K) and obtains squared median of all samples
    return np.median(np.sum((x[:,np.newaxis,:] - x[np.newaxis,:,:])**2, axis=-1))

# kernel and first and second order derivatives
def get_kernel(x):
    # receives sample of size (N,K) and returns the kernel with banwidth = squared median, as well as derivatives and trace
    gamma = get_gamma(x)

    # define rbf kernel
    def kernel(t1, t2): return np.exp(-0.5 * np.sum((t1 - t2)**2, axis = -1) / gamma) # returns (N,1)

    # derivatives of the kernel
    def dk_x(t1, t2): return egrad(lambda t : kernel(t, t2))(t1) # returns (N,K)
    def dk_y(t1, t2): return egrad(lambda t : kernel(t1, t))(t2) # returns (N,K)
    def dk_xy(t1, t2):
        # returns (N,1)
        g = 0
        for d in range(t1.shape[1]):
            g += egrad(lambda t : dk_x(t1,t)[:, d])(t2)[:, d]

        return g

    return kernel, dk_x, dk_y, dk_xy
