# Target distributions

Each script in this directory contains a different target density. The script actually defines multiple functions.

## Necessary function definitions

### Density functions
These functions are related to the target density.
* `logp_aux(x, K)` is the target log density (up to normalizing constant). `x` is a `shape(N,K)` array and `K` is the dimension of the density. Note that this function defines a family of densities - one for each possible dimension (although this parameter can be completely unused)
* `logp_ubvi(x)` is the target density used in UBVI. Usually, it can be a lambda function that simply calls `logp_aux`, but sometimes adjustments might be needed
* `sample(size, K)` generates the sample to initialize LBVI. `size` is an integer and `K` the dimension of the target density, with the same observations as before
* `p_sample(size, K)` generates a sample from the target density (if available). This is used to build a third-party KSD to compare the boosting VI methods

### LBVI schedules
These functions are related to LBVI weight optimization.
* `w_maxiters(k, long_opt)` determines for how many iterations the weight optimization is going to be run. `k` is an integer (iteration number) and `long_opt` is a boolean that indicates whether a long optimization should be done (if a new kernel is added to the mixture; can be ignored)
* `w_schedule(k)` determines the step sizes for SGD weight optimization. `k` is an integer (iteration number)

### UBVI schedules
These functions are related to UBVI component optimization.
* `adam_learning_rate(k)` determines the step size for ADAM in the component optimization. `k` is an integer (iteration number)
* `ubvi_gamma(k)` determines the step size for weight optimization. `k` is an integer (iteration number)

### BVI schedules
These functions are related to BVI optimizations.
* `gamma_alpha(k)` determines the step size for weight optimization via SGD. `k` is an integer (iteration number)
* `gamma_init(k)` determines the step size for new component initialization via SGD VI. `k` is an integer (iteration number)
* `regularization(k)` determines the regularization used to define the target in the new component optimization. `k` is an integer (iteration number)

## Directory roadmap
* `cauchy.py` contains the standard Cauchy distribution
* `banana.py` contains the banana distribution in 2 dimensions
* `doublebanana_gaussian.py` contains a mixture of a mixture of 4 bivariate Gaussians and a double banana in 2 dimensions
* `fivemixture.py` contains an evenly-spaced mixture of 5 Gaussians
* `fourmixture.py` contains a non-evenly-spaced mixture of 4 Gaussians. They are divided into two separate groups, each with two bumps
* `network.py` is used in the facebool-like network example
