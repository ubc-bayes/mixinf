# Tests

Each subdirectory (other than `targets`, `rkhs-kernels`, and `mixture-kernels`) contains a different test, which can be run by running the bash script. These tests run LBVI and BBBVI and compare the result in a single plot.

### Directory roadmap
* `targets` contains the targets that are approximated in the examples
* `rkhs-kernels` contains the RKHS kernels that are used to define the KSD
* `kernels` contains the kernels that are used to build the mixture
