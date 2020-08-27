import numpy as np

def weight_ard_active_fit_procedure(model, **kwargs):
    # fit weights w/o ard to initialize
    fit_args = dict(
        max_iter=50, update_active=False,
        update_weights=True, update_pi=True,
        ARD_weights=False
    )
    fit_args.update(kwargs)
    model.fit(**fit_args)

    # fit with ard
    fit_args = dict(
        max_iter=50, update_active=False,
        update_weights=True, update_pi=True,
        ARD_weights=True
    )
    fit_args.update(**kwargs)
    model.fit(**fit_args)

    # fit with active
    fit_args = dict(
        max_iter=1000, update_active=True,
        update_weights=True, update_pi=True,
        ARD_weights=True
    )
    fit_args.update(kwargs)
    model.fit(**fit_args)


def forward_fit_procedure(model, **kwargs):
    print('forward fit procedure')
    """
    for l in range(K):
        fit components 1...l

    fit all components jointly
    """
    for l in range(1, model.dims['K']):
        fit_args = dict(
            max_iter=1, update_active=False,
            update_weights=True, update_pi=True,
            ARD_weights=False, components=np.arange(l)
        )
        fit_args.update(**kwargs)
        print(fit_args)
        model.fit(**fit_args)

    fit_args = dict(
        max_iter=1000, update_active=True,
        update_weights=True, update_pi=True,
        ARD_weights=True
    )
    fit_args.update(**kwargs)
    print(fit_args)
    model.fit(**fit_args)

def fit_all(model, **kwargs):
    """
    standard fit procedure to run with good initialization
    update all parameters except tissue prection (not relevant to CAFEH-S)
    can change default arguments by passing **kwargs
    """
    fit_args = dict(
        max_iter=1000, update_active=True,
        update_weights=True, update_pi=True,
        ARD_weights=True
    )
    fit_args.update(**kwargs)
    print(fit_args)
    model.fit(**fit_args)
