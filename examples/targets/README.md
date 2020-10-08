# Target distributions

Each script in this directory contains a different target density. The script itself will define a log density function `p_aux(x, K)` and a sampling function `sample(size, K)`. Both functions depend on the dimension of the problem, and so in reality each script covers many target densities.

* each example will use one of these target distributions
* todo: add more options (eg banana dist)
