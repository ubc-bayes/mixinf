import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad

# kernel and first and second order derivatives
def get_kernel(k_gamma = 1):
    # t1,t2 are always (N,K) arrays

    # define rbf kernel
    def kernel(t1, t2): return np.exp(-0.5*np.sum((t1[:,np.newaxis,:] - t2[np.newaxis,:,:])**2, axis = -1)/k_gamma**2) # returns (N,N)

    # derivatives of the kernel
    def dk_x(t1, t2): return kernel(t1,t2)[:,:,np.newaxis] * (t2[:,np.newaxis,:] - t1[np.newaxis,:,:]) / k_gamma**2 # returns (N,N,K)
    def dk_y(t1, t2): return kernel(t1,t2)[:,:,np.newaxis] * (t1[:,np.newaxis,:] - t2[np.newaxis,:,:]) / k_gamma**2 # returns (N,N,K)
    def dk_xy(t1, t2): return kernel(t1,t2) * (1 - np.sum((t1[:,np.newaxis,:] - t2[np.newaxis,:,:])**2, axis = -1) / k_gamma**2) / k_gamma**2 # returns (N,N)

    return kernel, dk_x, dk_y, dk_xy
