# Hybrid AFQMC

This is the repo of corresponding code for the manuscript:
> Chen, Y., Zhang, L., E, W. & Car, R. (2022). Hybrid Auxiliary Field Quantum Monte Carlo for Molecular Systems. arXiv preprint [arXiv:2211.10824](https://arxiv.org/pdf/2211.10824.pdf).

You will need to use python and install `jax`, `flax`, `optax`, `pyscf`, `ml-collections` and `tensorboardX` to run the code. 

The [`hafqmc`](./hafqmc/) folder contains all the code and can be used directly as a package. Just make sure to add it to your `PYTHONPATH`. 

The [`examples`](./examples/) folder contains several examples that is shown in the manuscript. They can be directly run by something like `python run.py`. Note the `run_fprestart.py` will require you to run a optimization first and rename the `checkpoint.pkl` to `oldstates.pkl`.