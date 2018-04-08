import sys
import time
from copy import copy
import gpflow
import nlopt
import numpy as np
from pyDOE import lhs
from sklearn.metrics import mutual_info_score
from sampler.point_sampler import Point_Sampler
from sampler.slice_sampler import Slice_Sampler
from utilities.sampler_utilities import set_parameters_gp_bucb


class GP_BUCB(object):
    """Class that uses GP-BUCB to optimize function"""

    def __init__(self, parameters):
        """Initialize BO class"""

        self.t = 1
        self.parameters = copy(parameters)
        self.dim = copy(parameters['Dimension'])
        self.output_dim = 1
        self.input_dim = self.dim
        self.opt_dim = self.dim
        self.kernel_dim = self.input_dim
        self.opt_parameters = None
        self.X = None
        self.y = None
        self.X_var = None
        self.y_var = None
        self.gp_model = None
        self.batch_size = None
        self.X_train = None
        self.y_train = None
        self.init = 0
        self.train_size = 0
        self.kernel_number = 1
        self.set_params = set_parameters_gp_bucb  # can be replaced by to_dict
        self.beta = 3
        if 'Discrete_Mode' in parameters:
            self.discrete_mode = parameters['Discrete_Mode']
        else:
            self.discrete_mode = False
        if self.discrete_mode:
            self.discrete_set = parameters['Discrete_Set']
        if 'Parameters_Trained' in parameters:
            self.parameters_trained = parameters['Parameters_Trained']
        else:
            self.parameters_trained = False
        if 'Sampler' not in self.parameters:
            self.parameters['Sampler'] = 'PointSampler'

    def init_bo(self, batch_size, X_train, y_train):
        """Initialization for Bayesian Optimization"""

        if X_train is not None:
            self.X = copy(X_train)
            self.y = copy(y_train)
            self.X_var = copy(X_train)
            self.y_var = copy(y_train)
            self.train_size = X_train.shape[0]

        elif self.parameters_trained:
            self.X = np.zeros((1, self.input_dim))
            self.y = np.zeros((1, self.output_dim))
            self.X_var = np.zeros((1, self.input_dim))
            self.y_var = np.zeros((1, self.output_dim))
            self.init = 1
            self.train_size = 0

        else:
            self.X = np.zeros((batch_size, self.input_dim))
            samples = lhs(self.input_dim, batch_size, None)
            lower_bounds = np.matlib.repmat(
                np.asarray(self.parameters['Lower_Bounds']), batch_size, 1)
            upper_bounds = np.matlib.repmat(
                np.asarray(self.parameters['Upper_Bounds']), batch_size, 1)
            self.X = (upper_bounds - lower_bounds) * samples + lower_bounds
            self.y = np.zeros((batch_size, self.output_dim))
            self.X_var = copy(self.X)
            self.y_var = np.zeros((batch_size, self.output_dim))
            self.init = batch_size
            self.train_size = 0

        parameter_names = ['Lower_Bounds', 'Upper_Bounds', 'Xtol_Abs',
                           'Max_Iterations', 'Optimization_Algorithm']
        self.opt_parameters = copy(
            {k: self.parameters[k] for k in parameter_names})

        gp_model = self.build_model(copy(self.parameters), self.X, self.y)
        parameters_sampler = {
            'dim': self.kernel_dim,
            'fidelities': self.kernel_number,
            'Seed': self.parameters['Seed'],
            'Parameters_Trained': self.parameters['Parameters_Trained']}
        if self.parameters['Sampler'] == 'PointSampler':
            self.gp_model = Point_Sampler(parameters_sampler, gp_model,
                                          self.set_params)
        elif self.parameters['Sampler'] == 'SliceSampler':
            parameters_sampler.update(
                {'K': self.parameters['Iterations_Sampler'],
                 'Sigma_Kernel_Scale': self.parameters['Sigma_Kernel_Scale'],
                 'Noise_Scale': self.parameters['Noise_Scale'],
                 'Max_Lengthscale': self.parameters['Max_Lengthscale'],
                 'N': self.parameters['Samples_Sampler'],
                 'Simga_Kernel_Init': np.std(self.y)})
            self.gp_model = Slice_Sampler(parameters_sampler, gp_model,
                                          self.set_params)
        else:
            raise ValueError(self.parameters['Sampler'] +
                             ' is not a valid sampler')

        if not self.parameters_trained:
            self.gp_model.optimize(self.X, self.y)

        self.batch_size = batch_size

    def build_kernel(self, kernel_parameters):
        """Build Model Kernel"""

        active_dims = None
        if 'Active_Dims' in kernel_parameters:
            active_dims = kernel_parameters['Active_Dims']

        kernel = getattr(gpflow.kernels, kernel_parameters['Kernel'])(
            copy(kernel_parameters['Dimension']),
            variance=copy(kernel_parameters['Variance']),
            lengthscales=copy(kernel_parameters['Lengthscale']),
            active_dims=copy(active_dims),
            ARD=copy(kernel_parameters['ARD']))

        if kernel_parameters['Sampler'] == 'PointSampler':
            kernel = self.set_priors(kernel)

        if kernel_parameters['Kernel_Transform'] is not False:
            kernel.lengthscales.transform = gpflow.transforms.Logistic(
                kernel_parameters['Kernel_Transform'][0],
                kernel_parameters['Kernel_Transform'][1])

        return kernel

    def build_model(self, model_parameters, X, y):
        """Build GP Model"""

        zero_mean = False
        if model_parameters['Mean_Function'] == 'Zero':
            zero_mean = True
            model_parameters['Mean_Function'] = 0.0

        meanf = gpflow.mean_functions.Constant(
            model_parameters['Mean_Function'])

        kernel = self.build_kernel(model_parameters)

        model = gpflow.gpr.GPR(copy(X), copy(y), kern=copy(kernel),
                               mean_function=copy(meanf))
        if zero_mean:
            model.mean_function.c.fixed = True
        model.likelihood.variance = model_parameters['LH_Variance']

        return model

    def get_variance(self, x):
        """Get Variance of GP"""

        _, var = self.gp_model.predict(self.X_var, self.y_var, x)

        return var

    def get_mean(self, x):
        """Get mean of GP"""

        y, _ = self.gp_model.predict(self.X, self.y, x)

        return y

    def aquisition_function(self, x, grad):
        """Acquisition function of Bayesian Optimization"""

        x = np.reshape(x, (1, self.dim))
        std_dev = np.sqrt(self.get_variance(x)[0])
        mean = self.get_mean(x)[0]

        return mean + np.sqrt(self.beta) * std_dev


    def init_optimization(self):
        """Initialize optimizer with starting value"""

        initial_guess = []
        for i in range(self.dim):
            initial_guess.append((
                self.opt_parameters['Lower_Bounds'][i] +
                self.opt_parameters['Upper_Bounds'][i]) / 2)

        return initial_guess

    def optimize_function(self):
        """Optimize aquisition function using DIRECT method"""

        initial_guess = self.init_optimization()
        algorithm = getattr(
            nlopt, self.opt_parameters['Optimization_Algorithm'])

        opt = nlopt.opt(algorithm, self.opt_dim)
        opt.set_max_objective(self.aquisition_function)
        opt.set_lower_bounds(self.opt_parameters['Lower_Bounds'])
        opt.set_upper_bounds(self.opt_parameters['Upper_Bounds'])
        opt.set_xtol_abs(self.opt_parameters['Xtol_Abs'])
        opt.set_maxeval(self.opt_parameters['Max_Iterations'])

        x = opt.optimize(initial_guess)
        x = np.reshape(x, (-1, self.opt_dim))
        y = opt.last_optimum_value()

        return x, y

    def optimize_function_discrete(self):
        self.init_optimization()
        acq_value = np.zeros(self.discrete_set.shape[0])
        grad = None

        rows = self.discrete_set.shape[0]
        for row in range(rows):
            acq_value[row] =\
                self.aquisition_function(self.discrete_set[row, :self.opt_dim],
                                         grad)

        idx = np.argmax(acq_value)
        x = np.reshape(self.discrete_set[idx, :self.opt_dim],
                       (-1, self.opt_dim))
        y = acq_value[idx]

        return x, y

    def get_new_beta(self):

        max_C = 0
        a = 1.0
        b = 1.0
        r = 0.75
        d = 1.0

        if self.batch_size > 1:
            choice = np.asarray(range(0, self.X.shape[0]))
            for idx in range(100):
                indices = np.random.choice(choice, size=self.batch_size - 1,
                                           replace=True)
                ys = np.take(self.y, indices, axis=0)
                xs = np.take(self.X, indices, axis=0)
                f, _ = self.m.predict(self.X, self.y, xs)
                c_xy = np.histogram2d(f, ys[:, 0], 100)[0]
                C = mutual_info_score(None, None, contingency=c_xy)
                if C > max_C:
                    max_C = C

        d_size = self.X.shape[1]

        t = int(self.t / self.batch_size) * self.batch_size
        beta = (2 * np.log(t**2 * 2 * np.pi ** 2 / (3 * d)) +
                2 * d_size * np.log(t**2 * d_size * b * r *
                                    np.sqrt(np.log(4 * d_size * a / d)))) *\
            np.exp(2.0 * max_C)
        return beta

    def get_new_sample_points(self):
        """Return new sample points"""

        if self.parameters['Adaptive_Beta'] and (self.train_size > 0 or
                                                 self.t > 1):
            try:
                self.beta = self.get_new_beta()
            except ValueError:
                self.beta = 3.0
        else:
            self.beta = 3.0

        for batch_number in range(self.batch_size - self.init):
            if self.discrete_mode:
                x, _ = self.optimize_function_discrete()
            else:
                x, _ = self.optimize_function()
            self.X_var = np.vstack((self.X_var, x))
            self.y_var = np.zeros((self.X_var.shape[0], self.output_dim))

        return copy(self.X_var[(-1 * self.batch_size):, :])

    def update_model(self, X_new, y_new):
        """Update GP with new function values"""

        y_new = np.reshape(y_new, (-1, self.output_dim))

        if self.init > 0:
            self.y = y_new
            self.X = X_new
            self.init = 0
        else:
            self.y = np.vstack((self.y, y_new))
            self.X = np.vstack((self.X, X_new))

        self.t = self.t + 1

        if self.parameters['Update_Parameters'] is True:
            self.update_model_parameters()

    def update_model_parameters(self):
        """Retrain the GP model with newly gained information"""

        self.gp_model.optimize(self.X, self.y)

    def get_bo_history(self):
        """Return the Bayesian Optimization History"""

        return copy(self.X[self.train_size:, :]),\
            copy(self.y[self.train_size:, :]), \
            copy(self.y[self.train_size:, :])

    def optimize_model(self, x, y, kernel_list):
        """Train GP model"""

        results = {}
        for mean_function in self.parameters['Mean_Functions']:

            self.parameters['Mean_Function'] = mean_function
            if mean_function == 'Constant':
                self.parameters['Mean_Function'] = 0.0

            for k in kernel_list:
                if 'Lengthscale' not in self.parameters:
                    self.parameters['Lengthscale'] = None
                self.parameters['Kernel'] = k
                model = self.build_model(self.parameters, copy(x), copy(y))
                model.optimize()
                results = self.get_optimized_parameters(model, results,
                                                        mean_function, k)

        max_likelihood = -float('Inf')
        max_kernel = ''
        for kernel, parameters in results.iteritems():
            likelihood = parameters['Optimized_Likelihood']
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_kernel = kernel

        return max_kernel, results[max_kernel]

    def get_kernel_parameters(self, model, parameters):
        """Get kernel parameters"""

        parameters['Optimized_Variance'] = \
            model.kern.variance.value.tolist()
        parameters['Optimized_Lengthscales'] = \
            model.kern.lengthscales.value.tolist()

        return parameters

    def set_priors(self, kernel):
        kernel.lengthscales.prior = gpflow.priors.LogNormal(
            self.parameters['Sampler_Prior_Mean'],
            self.parameters['Sampler_Prior_Variance'])
        return kernel

    def get_optimized_parameters(self, model, results, mean_function, kernel):
        """Get other optimized parameters"""

        parameters = {}
        parameters = self.get_kernel_parameters(model, parameters)
        parameters['Optimized_LH_Variance'] = \
            model.likelihood.variance.value.tolist()
        parameters['Optimized_Likelihood'] = \
            model.compute_log_likelihood()

        if mean_function == 'Constant':
            parameters['Optimized_Mean_Function'] = \
                model.mean_function.c.value.tolist()
        else:
            parameters['Optimized_Mean_Function'] = ['Zero']

        if isinstance(kernel, tuple):  #do better
            kernel_name = ''
            for k in kernel:
                kernel_name = kernel_name + '_' + k
            results[kernel_name + '_' + mean_function] = parameters
        else:
            results[kernel + '_' + mean_function] = parameters

        return results

    def set_batch_size(self, batch_size):
        """Set new batch size"""

        self.batch_size = batch_size
