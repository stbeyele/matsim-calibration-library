import sys
import gpflow


def set_parameters_gp_bucb(hyperparameters, model):
    model.kern.lengthscales = hyperparameters['lengthscales']
    model.kern.variance = hyperparameters['kernel_variance']
    model.likelihood.variance = hyperparameters['lh_variance']
    model.mean_function.c = hyperparameters['mean']
    return model


def set_parameters_gp_bucb_dim(hyperparameters, model):
    kernels = model.kern.sorted_params
    for kernel in kernels:
        kernel_name = kernel.name
        kernel_object = getattr(model.kern, kernel_name)
        idx = kernel_object.active_dims
        kernel_object.lengthscales =\
            hyperparameters['lengthscales'][idx]
        kernel_object.variance =\
            hyperparameters['kernel_variance'][idx]
    model.likelihood.variance = hyperparameters['lh_variance']
    model.mean_function.c = hyperparameters['mean']
    return model


def set_parameters_mes_fidelity(hyperparameters, model):
    kernels = model.kern.sorted_params
    dim = hyperparameters['dim']

    for idx, kernel in enumerate(kernels):
        kernel_name = kernel.name
        kernel_object = getattr(model.kern, kernel_name)
        if isinstance(kernel, gpflow.kernels.Prod):
            prod_kernel = kernel_object.sorted_params[1]
            kernel_object = getattr(kernel_object, prod_kernel.name)
        kernel_object.lengthscales =\
            hyperparameters['lengthscales'][idx * dim: (idx + 1) * dim]
        kernel_object.variance =\
            hyperparameters['kernel_variance'][idx]
    model.likelihood.variance = hyperparameters['lh_variance']
    model.mean_function.c = hyperparameters['mean']
    return model


def set_priors_gp_bucb(model):
    model.kern.lengthscales.prior = gpflow.priors.Gamma(1, 1)
    model.kern.variance.prior = gpflow.priors.Gamma(1, 1)
    model.likelihood.variance.prior = gpflow.priors.Gamma(1, 1)
    model.mean_function.c.prior = gpflow.priors.Gaussian(0, 10)
    return model
