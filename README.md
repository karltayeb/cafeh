# CAFEH: Colocalization and Finemapping in the presence of Allelic Heterogeniety

This repository contains code for fitting and plotting results of CAFEH model. CAFEH is a method that performs finemapping and colocalization jointly over multiple phenotypes. It can support

Please refer to `notebooks/CAFEH_demo.ipynb` for an example of how to use CAFEH

### Install

To install CAFEH, clone the repository and install via pip

```
git clone https://github.com/karltayeb/cafeh.git
cd cafeh
pip install .
```

### Important files in this repository

`cafeh/cafeh_summary.py`: code for class `CAFEHSummary` for fitting CAFEH with summary stats
`cafeh/cafeh_genotype.py`: code for class `CAFEHGenotype` for fitting CAFEH with individual level data
`cafeh/cafeh.py`: high level scripts for running CAFEH in one command

`cafeh/cafeh_model_queries.py`: methods for querying `CAFEHSummary` and `CAFEHGenotype` objects
`cafeh/plotting.py`: some useful plots for CAFEH

`notebooks/CAFEH_demo.ipynb`: Simple example running CAFEH Genotype and CAFEH Summary


### How to use

```
from cafeh.cafeh import fit_cafeh_genotype
from cafeh.fitting import weight_ard_active_fit_procedure

# initialize and fit model
cafehg = fit_cafeh_genotype(X, y, K=10)
```



