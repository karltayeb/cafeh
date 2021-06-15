# CAFEH: Colocalization and Finemapping in the presence of Allelic Heterogeniety

This repository contains code for fitting and plotting results of CAFEH model. CAFEH is a method that performs finemapping and colocalization jointly over multiple phenotypes. CAFEH can be run with 10s of phenotypes and 1000s of variants in a few minutes.

Please refer to `notebooks/CAFEH_demo.ipynb` for an example of how to use CAFEH

## Install

```
git clone https://github.com/karltayeb/cafeh.git
cd cafeh
conda env create --file environment.yml  # create envrionment with dependencies
conda activate cafeh  # activate environment
pip install .  # install package in cafeh environment
```

## How to use

You can use CAFEH as a command line tool or a python package. Read below for a minimum example of how to use CAFEH from the command line. Look at `notebooks/CAFEH_demo.ipynb` for an example of how to interact with CAFEH in Python.

### Command line:

To use CAFEH from the command line your input data will need to be formatted as a tab delimited text file. If you are using individual level data you will need to provide genotypes, phenotypes, and additional covariates. If you are running CAFEH with summary stats you will need to provide a reference LD matrix AND (effect sizes + standard errors OR z scores) AND sample sizes. For examples of how this input should be formatted look in the `example` subdirectory

To run CAFEH genotype, include `--mode genotype` in the arguments. You must specify `--genotype (-X)`, `--phenotype (-Y)`, if there are other covariates you may include them with `--covariates (-c)`

The output will be saved to the directory specified by `--out` flag, in the example below, output is saved to a directory `output`

```
python cafeh.py --mode genotype -X example/X_example.tsv -Y example/y_example.tsv -c example/cov_example.tsv --out output --save-model                                                                                          current_working_branch ✱ ◼
fitting CAFEH genotype...
saving output...
saving results table to output/cafeh.genotype.results
saving cafeh model to output/cafeh.genotype.model
```

To run CAFEH with effect sizes and standard errors, include `--mode beta` in the arguments. You must provide input `--ld (-R)`, `--betas (-B)`, `--standard-errors (-S)`, and `--sample-sizes (-S)`.

```
python cafeh.py --mode beta -R example/LD_example.tsv -B example/beta_example.tsv -S example/stderr_example.tsv -n example/n_example.tsv --out output --save-model                                                              current_working_branch ✱ ◼
fitting CAFEH with effect sizes and standard errors...
saving output...
saving results table to output/cafeh.beta.results
saving cafeh model to output/cafeh.beta.model
```

To fit CAFEH with zscores, use `--mode z` in the arguments. You must provide input `--ld (-R)`, and `--zscores (-z)`. In lieu of providing z scores you can also use the effect sizes and standard errors as input via `-B` and `-S`. The script will use these to compute z scores.


```
python cafeh.py --mode z -R example/LD_example.tsv -z example/z_example.tsv -n example/n_example.tsv --out output --save-model                                                                                                  current_working_branch ✱ ◼
fitting CAFEH with z scores...
saving output...
saving results table to output/cafeh.z.results
saving cafeh model to output/cafeh.z.model
```


For a full list of options run

```
python cafeh.py -h
```

### CAFEH output:

After you run the script there will be two files saved to the output directory

```
output
├── cafeh.{mode}.model
└── cafeh.{mode}.results
```

The `.model` file is a pickle of the CAFEH model. You can load this pickle and interact with the CAFEH model in python. To generate this file you must include the `--save-model` flag.

The `.results` model provides a useful summary of CAFEH's output. It contains a row for each (SNP, study) pair. It reports
- `pip` The posterior inclusion probability for that SNP in that study
- `top_component` the CAFEH component with the largest probability for that SNP
- `alpha` the smallest credible set level at which the SNP would be included in the credible set for `top_component`. e.g `alpha = 0.3` means that the SNP would be included in every `a * 100 %` credible set with `a >= alpha`
-`rank` is the rank of the SNP in `top_component`. A rank of `n` indicates that there are `n` SNPs with larger posterior probability in `top_component`
- `p_active` gives the probability that `top_component` is an active component in this study
- `pi` is the posterior probability of the SNP being the causal SNP for `top_component`. This is distinct from `pip` which is the probability of this SNP being causal in any component
- `effect` and `effect_var` are the posterior mean and variance of the effect size for this SNP in this study, conditioned on this SNP being the causal SNP.


### Python:
```
from cafeh.cafeh import fit_cafeh_genotype
from cafeh.fitting import weight_ard_active_fit_procedure

# prepare inputs for cafeh
...

# initialize and fit model
cafehg = fit_cafeh_genotype(X, y, K=10)

# downstream analysis of model fitting
...

```


### Important files in this repository

`cafeh/cafeh_summary.py`: code for class `CAFEHSummary` for fitting CAFEH with summary stats
`cafeh/cafeh_genotype.py`: code for class `CAFEHGenotype` for fitting CAFEH with individual level data
`cafeh/cafeh.py`: high level scripts for running CAFEH in one command

`cafeh/cafeh_model_queries.py`: methods for querying `CAFEHSummary` and `CAFEHGenotype` objects
`cafeh/plotting.py`: some useful plots for CAFEH

`notebooks/CAFEH_demo.ipynb`: Simple example running CAFEH Genotype and CAFEH Summary





