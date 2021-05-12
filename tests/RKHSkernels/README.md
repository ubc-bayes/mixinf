# RKHS kernels

Each script in this directory contains a different RKHS kernel. Each script should define four functions:

* `kernel(x,y)`, the actual kernel. It should receive two `shape(N,)` arrays and return a `shape(N,1)` array
* `dk_x(x,y)`, the partial derivative wrt to x of the kernel. It should receive two `shape(N,)` arrays and return a `shape(N,K)` array, where `K` is the dimension of the problem
* `dk_y(x,y)`, the partial derivative wrt to y of the kernel. It should receive two `shape(N,)` arrays and return a `shape(N,K)` array, where `K` is the dimension of the problem
* `dk_xy(x,y)`, the trace of the hessian of the kernel. It should receive two `shape(N,)` arrays and return a `shape(N,1)` array

### Directory roadmap
* `rbf.py` contains an RBF kernel; the bandwidth can be adjusted in the script
* `rbf_median.py` contains another RBF kernel. Specifically, the functions in this script receive a `shape(N,K)` array and generate an RBF where the bandwidth is the median of the squared distances between all elements of the array. This script is used to construct a third-party KSD to compare LBVI with different boosting VI methods
