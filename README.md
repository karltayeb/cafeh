# CAFEH: Colocalization and Finemapping in the presence of Allelic Heterogeniety

code for fitting, evaluating and plotting results of CAFEH model

To run CAFEH Summary
```
from cafeh.cafeh_ss import CAFEH
from cafeh.fitting import weight_ard_active_fit_procedure
# N variants, M samples, T studies
# R: [N x N] LD matrix (pairwise correlation of all varaints in model)
# B: [T x N] matrix of linear regression effect sizes
# S: [T x N] matrix of S_tn = Var(y_t) / (Var(x_n) * M), for large M approximate with Var(beta_tn)

# initialize model
model = CAFEH(R, B, S, K=K)

# fit model
# we have implemented a few other optimization schemes, 'forward_fit_procedure' and 'fit_all'
weight_ard_active_fit_procedure(model)

# save model
# by default we do not save the data with the model, if you want the data set save_data=True
model.save({path}, save_data=True)

# load model
# we save a compact version of the model, when you load the model call _decompress_model()

import pickle
model = pickle.load(open({path}, 'rb'))
model._decompress_model()

# plot_model
from cafeh.misc import plot_components
plot_components(model)
```
