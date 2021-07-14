# Locally-adaptive boosting variational inference

The main code to do LBVI is in this folder, along with code for boosting black box VI (Locatello et al. 2018a,b), universal VI (Campbell and Li, 2019), and Hamiltonian Monte Carlo (Neal, 2011).

### Directory roadmap

* `lbvi.py` contains the suite of functions for LBVI, where the steps of a different component are increased greedily
* `lbvi_smc.puy` contains the suite of functions for LBVI with SMC components, which uses the KL divergence to greedily estimate the target
* `uniform_lbvi.py` contains the suite of functions for uniform LBVI, where the steps of all active components are increased in each iteration
* `bvi.py` contains the suite of functions for black box VI
* `ubvi.py` contains the suite of functions for universal VI (taken from the [authors' Github](https://github.com/trevorcampbell/ubvi) and modified as needed)
* `hmc.py` contains the code for running HMC
