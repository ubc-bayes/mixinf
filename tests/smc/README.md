# SMC mixture components

Each script in this directory contains a different SMC mixture component. Specifically, each scripts should receive parameters (e.g. about the Markov kernel) and return a function `smc(logp, logr, r_sample, B, beta_ls, Z0)` via a `create_smc(params)` function.

Here,
* `logp` is the target log density
* `logr` is the reference log density
* `r_sample` generates samples from the reference distribution
* `B` is the number of particles to generate (i.e. output sample size)
* `beta_ls` is the grid of beta values
* `Z0` is the normalizing constant of the reference density

### Directory roadmap
* `smc.py` contains an SMC sampler with RWMH rejuvenation kernels
