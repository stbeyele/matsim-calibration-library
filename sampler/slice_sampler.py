import numpy as np
import sys
import gpflow
from copy import copy, deepcopy
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from numpy.linalg.linalg import LinAlgError
import scipy.linalg
#  import numpy.random as npr
#  import scipy.linalg as spla



class Slice_Sampler(object):

    def __init__(self, parameters, model, set_parameters):
        print parameters
        self.N = parameters['N']
        self.dim = parameters['dim']
        self.fidelities = parameters['fidelities']
        self.noiseless = False
        self.sigma_kernel_scale = parameters['Sigma_Kernel_Scale']  # 1.0
        self.noise_scale = parameters['Noise_Scale']  # horseshoe prior 0.1
        self.max_ls = parameters['Max_Lengthscale']
        self.model = model
        self.set_parameters = set_parameters
        self.K = parameters['K']
        self.seed = parameters['Seed']
        self.lh_logger = []
        self.hyperparameters = None
        self.trained_models = None
        self.N_opt = None
        self.parameters_logger = []
        self.working_models = None

        hyperparameters_init = {}
        hyperparameters_init['lengthscales'] =\
            np.ones((self.dim * self.fidelities))
        hyperparameters_init['kernel_variance'] =\
            (parameters['Simga_Kernel_Init'] + 1e-4) *\
            np.ones((self.fidelities))
        hyperparameters_init['lh_variance'] = 1e-3
        hyperparameters_init['mean'] = 0.0
        hyperparameters_init['dim'] = self.dim
        self.hyperparameters_last_it = copy(hyperparameters_init)
        self.hyperparameters_init = copy(hyperparameters_init)

    def dir_logprob(self, z, direction, init_x, logprob, X, y,
                    hyperparameters):
        return logprob(direction * z + init_x, X, y, hyperparameters)

    def direction_slice(self, direction, init_x, sigma, step_out,
                        max_steps_out, logprob, X, y, hyperparameters):

        upper = sigma * np.random.rand()
        lower = upper - sigma
        llh_s = np.log(np.random.rand()) +\
            self.dir_logprob(0.0, direction, init_x, logprob, X, y,
                             hyperparameters)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while self.dir_logprob(lower, direction, init_x, logprob, X, y,
                                   hyperparameters) > llh_s\
                and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower -= sigma
            while self.dir_logprob(upper, direction, init_x, logprob, X, y,
                                   hyperparameters) > llh_s\
                and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper += sigma

        steps_in = 0
        while True:
            steps_in += 1
            new_z = (upper - lower) * np.random.rand() + lower
            new_llh = self.dir_logprob(new_z, direction, init_x, logprob, X,
                                       y, hyperparameters)
            if np.isnan(new_llh):
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z * direction + init_x

    def slice_sample(self, init_x, logprob, X, y, hyperparameters, sigma=1.0,
                     step_out=True, max_steps_out=1000, compwise=False):

        if isinstance(init_x, list):
            init_x = np.asarray(init_x)

        dims = init_x.shape[0]
        if compwise:
            ordering = range(dims)
            np.random.shuffle(ordering)
            cur_x = init_x.copy()
            for d in ordering:
                direction = np.zeros((dims))
                direction[d] = 1.0
                cur_x = self.direction_slice(direction, cur_x, sigma, step_out,
                                             max_steps_out, logprob, X, y,
                                             hyperparameters)
            return_value = cur_x

        else:
            direction = np.random.randn(dims)
            direction = direction / np.sqrt(np.sum(direction**2))
            return_value = self.direction_slice(direction, init_x, sigma,
                                                step_out, max_steps_out,
                                                logprob, X, y, hyperparameters)
        return return_value

    def sample_ls(self, X, y, hyperparameters):
        ls = self.slice_sample(hyperparameters['lengthscales'],
                               self.logprob_ls, X, y, copy(hyperparameters),
                               compwise=True)
        hyperparameters['lengthscales'] = ls
        return hyperparameters

    def logprob_ls(self, ls, X, y, hyperparameters):

        kernel_parameters = copy(hyperparameters)
        kernel_parameters['lengthscales'] = ls

        if np.any(ls < -10) or np.any(ls > self.max_ls):
            return -float("inf")

        try:
            # self.model.X = X
            # self.model.Y = y
            # self.model = self.set_parameters(kernel_parameters, self.model)
            # self.check_if_chol_possible(X, self.model, kernel_parameters)
            # lp = self.model.compute_log_likelihood()
            lp = self.calc_lh(X, y, kernel_parameters)
        except LinAlgError:
            return -float("inf")
        except ValueError:
            return -float("inf")
        return lp

    def calc_lh(self, X, y, hyperparameters):

        self.model.X = X
        self.model.Y = y
        self.model = self.set_parameters(hyperparameters, self.model)

        cov = self.model.kern.compute_K_symm(X) +\
            hyperparameters['lh_variance'] * np.eye(X.shape[0])
        #  cov = np.nan_to_num(cov)
        chol = scipy.linalg.cholesky(cov, lower=True)
        solve = scipy.linalg.cho_solve((chol, True),
                                       y - hyperparameters['mean'])
        lp = -np.sum(np.log(np.diag(chol))) - 0.5 *\
            np.dot((y - hyperparameters['mean'])[:, 0], solve[:, 0])

        return lp

    def logprob_noisy(self, parameters, X, y, hyperparameters):

        kernel_parameters = copy(hyperparameters)
        kernel_parameters['kernel_variance'] = parameters[0:self.fidelities]
        kernel_parameters['lh_variance'] = parameters[self.fidelities]
        kernel_parameters['mean'] = parameters[self.fidelities + 1]

        # if (kernel_parameters['mean'] > np.max(y) or
        #     kernel_parameters['mean'] < np.min(y)):
        #     return -float("inf")

        #if kernel_parameters['mean'] < 0:
        #    return -float("inf")

        if (np.any(kernel_parameters['kernel_variance'] < 0) or
            kernel_parameters['lh_variance'] < 0):
            return -float("inf")

        try:
            # self.model.X = X
            # self.model.Y = y
            # self.model = self.set_parameters(kernel_parameters, self.model)
            # self.check_if_chol_possible(X, self.model, kernel_parameters)
            # lp = self.model.compute_log_likelihood()
            lp = self.calc_lh(X, y, kernel_parameters)
        except LinAlgError:
            return -float("inf")

        # Roll in noise horseshoe prior.
        lp += np.log(np.log(1 + (self.noise_scale /
                                 kernel_parameters['lh_variance'])**2))

        # Roll in amplitude lognormal prior
        lp -= 0.5 * np.sum((np.log(kernel_parameters['kernel_variance']) /
                            self.sigma_kernel_scale)**2)

        return lp

    def sample_noisy(self, X, y, hyperparameters):
        hypers = self.slice_sample(
            np.hstack((
                hyperparameters['kernel_variance'],
                hyperparameters['lh_variance'],
                hyperparameters['mean'])),
            self.logprob_noisy,
            X, y, copy(hyperparameters), compwise=False)

        hyperparameters['kernel_variance'] = hypers[0:self.fidelities]
        hyperparameters['lh_variance'] = hypers[self.fidelities]
        hyperparameters['mean'] = hypers[self.fidelities + 1]
        return hyperparameters

    def sample_hypers(self, X, y, hyperparameters):
        hyperparameters = self.sample_noisy(X, y, copy(hyperparameters))
        hyperparameters = self.sample_ls(X, y, copy(hyperparameters))
        return hyperparameters

    def optimize(self, X, y):

        hyperparameters_optimized = []
        hyperparameters = deepcopy(self.hyperparameters_last_it)
        print '------------------new iteration------------------'
        print hyperparameters
        np.random.seed(self.seed)

        for n in range(self.N):
            for k in range(self.K):
                hyperparameters = self.sample_hypers(X, y, hyperparameters)
            hyperparameters_optimized.append(hyperparameters)
        self.parameters_logger.append(hyperparameters_optimized)
        self.hyperparameters = hyperparameters_optimized
        print self.hyperparameters
        self.hyperparameters_last_it = hyperparameters_optimized[-1]
        self.trained_models = []
        for parameters in hyperparameters_optimized:
            trained_model = deepcopy(self.model)
            trained_model = self.set_parameters(parameters,
                                                trained_model)
            self.trained_models.append(trained_model)
            # try:
            #     self.check_if_chol_possible(X, trained_model, parameters)
            #     self.trained_models.append(trained_model)
            # except LinAlgError:
            #     print 'Found Parameters not working!!!!'

        self.lh_logger.append(self.calc_final_lh(X, y))
        self.N_opt = len(self.trained_models)

    def check_if_chol_possible(self, X, model, hyperparameters):
        cov = model.kern.compute_K_symm(X) +\
            hyperparameters['lh_variance'] * np.eye(X.shape[0])
        chol = scipy.linalg.cholesky(cov, lower=True)


    def calc_final_lh(self, X, y):

        N = len(self.trained_models)
        lh = np.zeros((N))
        self.model.X = X
        self.model.Y = y

        for n in range(N):

            model = self.trained_models[n]

            model.X = X
            model.Y = y

            lh[n] = model.compute_log_likelihood()

            # Roll in noise horseshoe prior.
            lh[n] += np.log(np.log(1 + (
                self.noise_scale / self.hyperparameters[n]['lh_variance'])**2))

            # Roll in amplitude lognormal prior
            lh[n] -= 0.5 * np.sum((np.log(
                self.hyperparameters[n]['kernel_variance']) /
                self.sigma_kernel_scale)**2)

        return np.mean(lh)

    def get_lh_history(self):
        return deepcopy(self.lh_logger)

    def get_parameters_history(self):
        return deepcopy(self.parameters_logger)

    def predict(self, X, y, X_pred):

        mu, s2, _ = self.predict_samples(X, y, X_pred)

        fmu = np.mean(mu, 1)
        fs2 = np.mean(s2, 1)

        return fmu, fs2

    def predict_samples(self, X, y, X_pred):
        N = len(self.trained_models)
        rows = X_pred.shape[0]
        fmu = np.zeros((rows, N))
        fs2 = np.zeros((rows, N))

        for n in range(N):
            model = self.trained_models[n]
            model.X = X
            model.Y = y
            mu, s2 = model.predict_y(X_pred)
            fmu[:, n] = mu[:, 0]
            fs2[:, n] = s2[:, 0]
        weights = np.ones(N)

        return fmu, fs2, weights

    def predict_sample(self, X, y, X_pred, n):

        model = self.trained_models[n]
        model.X = X
        model.Y = y
        try:
            self.check_if_chol_possible(X_pred, model, self.hyperparameters[n])
            mu, s2 = model.predict_y(X_pred)
            fmu = mu[:, 0]
            fs2 = s2[:, 0]

        except LinAlgError:
            model = deepcopy(self.working_models[n])
            self.trained_models[n] = model
            mu, s2 = model.predict_y(X_pred)
            fmu = mu[:, 0]
            fs2 = s2[:, 0]

        self.working_models = deepcopy(self.trained_models)

        return fmu, fs2, 1.0

    def get_lh_variance(self, idx):

        #return np.absolute(self.hyperparameters[idx]['lh_variance'])
        return self.hyperparameters[idx]['lh_variance']

    def get_N(self):
        return self.N_opt


