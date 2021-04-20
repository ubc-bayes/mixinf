# Target distributions

Each script in this directory contains a different target density. The script itself will define a log density function `logp_aux(x, K)`, a score function `sp(x, K)`,  and a sampling function `sample(size, K)`. Both functions depend on the dimension of the problem, and so in reality each script covers many target densities.

* each example will use one of these target distributions
