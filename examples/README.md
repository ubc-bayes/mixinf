# Self-contained examples

Each directory contains an example with a specific target distribution. To run a specific example, navigate to its directory and run the `run.sh` bash script.

### Directory roadmap
* `experiment.py` is the main code used to run each experiment. The bash files all run this script, and modify the arguments that it receives
* `univariate-plot.py` produces plots for univariate target densities
* `bivariate-plot.py` produces plots for bivariate target densities
* `targets` contains the targets that are approximated in the examples
* Folders starting with `seq` use the sequential algorithm, `nsvmi`
* Folders ending in `lp` use a linear program to optimize the weights, and are all sequential (i.e. use `nsvmi`)
