import sys
from copy import copy
import itertools
import gpflow
from bayesian_optimizer.gp_bucb import GP_BUCB
from utilities.sampler_utilities import set_parameters_gp_bucb_dim


class GP_BUCB_Dimension(GP_BUCB):
    """Implements GP-BUCB with a separate Kernel for each Dimension"""

    def __init__(self, parameters):
        """Set specific input and output dimensions and init add. variable"""
        super(self.__class__, self).__init__(parameters)
        self.set_params = set_parameters_gp_bucb_dim
        self.kernel_number = self.input_dim
        self.kernel_dim = 1

    def build_kernel(self, model_parameters):
        """Build the kernel"""

        kernel_list = model_parameters['Kernel']

        kernel = copy(getattr(gpflow.kernels, kernel_list[0])(
            1,
            variance=model_parameters['Variance'][0],
            lengthscales=model_parameters['Lengthscale'][0],
            active_dims=[0],
            ARD=False))

        if model_parameters['Sampler'] == 'PointSampler':
            kernel = self.set_priors(kernel)

        kernel_total = kernel
        for i in range(1, len(kernel_list)):
            kernel = copy(getattr(gpflow.kernels, kernel_list[i])(
                1,
                variance=model_parameters['Variance'][i],
                lengthscales=model_parameters['Lengthscale'][i],
                active_dims=[i],
                ARD=False))
            if model_parameters['Sampler'] == 'PointSampler':
                kernel = self.set_priors(kernel)
            kernel_total = kernel_total * kernel

        return kernel_total

    def optimize_model(self, x, y, kernel_list):
        """ Train GP model"""

        combination_list = []
        kernel_combinations = []
        for idx in range(self.dim):
            combination_list.append(kernel_list) #better with multiplication combination_list = kernel_list * self.dim
        for combination in itertools.product(*combination_list):
            kernel_combinations.append(combination)

        self.parameters['Lengthscale'] = None #really needed???
        for kernel_parameter in ['Variance', 'Lengthscale']:
            self.parameters[kernel_parameter] = self.dim * \
                [self.parameters[kernel_parameter]]

        max_kernel, result = super(self.__class__, self).optimize_model(
            x, y, kernel_combinations)

        return max_kernel, result

    def get_kernel_parameters(self, model, parameters):
        """Get kernel parameters"""

        kernels = model.kern.sorted_params

        for kernel in kernels:
            kernel_parameters = {}
            kernel_parameters['Optimized_Dimension'] = \
                kernel.active_dims.tolist()[0]
            kernel_parameters['Optimized_Variance'] = \
                kernel.variance.value.tolist()[0]
            kernel_parameters['Optimized_Lengthscales'] = \
                kernel.lengthscales.value.tolist()[0]
            parameters[kernel.name] = kernel_parameters

        return parameters

    def get_optimized_parameters(self, model, results, mean_function,
                                 kernel_combination):
        """Get optimized parameters"""

        kernel_name = ''
        for kernel in kernel_combination:
            kernel_name = kernel_name + kernel

        results = super(self.__class__, self).get_optimized_parameters(
            model, results, mean_function, kernel_name)

        return results
