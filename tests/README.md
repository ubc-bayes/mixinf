# Tests

Test subdirectories run LBVI and other VI and MCMC routines and compare the results. Each test can be run by executing the bash script in the corresponding subdirectory. Each bash script will execute one or two of the scripts, which we detail below. Auxiliary subdirectories contain information about the target distribution, the tuning parameters of each method, and RKHS kernels used for the KSD, and the RWMH kernels used in the mixture.

### Tests roadmap
* `cauchy1d` contains tests on the standard, univariate Cauchy distribution
* `banana-gaussian` contains tests on a mixture of four bivariate Gaussians and a double banana distribution

### Scripts roadmap
* `tests.py` contains the code to run a generic test. Type `python3 tests.py --help` to see help, or run one of the test bash scripts
* `univariate-plot.py` contains the code to plot the results for tests with univariate distributions. Type `python3 univariate-plot.py --help` to see help
* `bivariate-plot.py` contains the code to plot the results for tests with bivariate distributions. Type `python3 bivariate-plot.py --help` to see help

### Auxiliary subdirectories
* `targets` contains the targets that are approximated in the examples
* `RKHSkernels` contains the RKHS kernels that are used to define the KSD
* `kernels` contains the kernels that are used to build the mixture
