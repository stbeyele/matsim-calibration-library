import gpflow
import sys
from copy import deepcopy
import numpy as np
from tensorflow.python.framework.errors_impl import InvalidArgumentError

class Point_Sampler(object):

    def __init__(self, parameters, model, set_parameters):

        print parameters
        self.model = model
        self.set_parameters = set_parameters
        self.dim = parameters['dim']
        self.seed = parameters['Seed']
        self.lh_logger = []
        self.parameters_logger = []

        if not parameters['Parameters_Trained']:
            mean = 0.0
            lh_variance = 1e-3
            kernel_variance = np.ones((parameters['fidelities']))
            lengthscales = np.ones((parameters['dim'] *
                                    parameters['fidelities']))

            kernel_parameters = {'lengthscales': lengthscales,
                                 'kernel_variance': kernel_variance,
                                 'lh_variance': lh_variance,
                                 'mean': mean,
                                 'dim': self.dim}
            self.model = self.set_parameters(kernel_parameters, self.model)

    def predict(self, X, y, X_pred):

        self.model.X = X
        self.model.Y = y
        fmu, fs2 = self.model.predict_y(X_pred)
        return fmu[:, 0], fs2[:, 0]

    def predict_sample(self, X, y, X_pred, idx):
        fmu, fs2 = self.predict(X, y, X_pred)
        return fmu, fs2, 1.0

    def predict_samples(self, X, y, X_pred):
        fmu, fs2 = self.predict(X, y, X_pred)
        fmu = np.reshape(fmu, (1, -1))
        fs2 = np.reshape(fs2, (1, -1))
        return fmu, fs2, np.ones(1)


    def optimize(self, X, y):
        np.random.seed(self.seed)

        self.model.X = X
        self.model.Y = y

        parameters = self.model.get_parameter_dict()

        try:
            self.model.optimize()
        except InvalidArgumentError:
            self.model.set_parameter_dict(parameters)
            print 'using last parameter configuration'

        self.lh_logger.append(self.calc_final_lh(X, y))
        self.parameters_logger.append(self.model.get_parameter_dict())

        print self.model

    def get_lh_variance(self, idx):
        return self.model.likelihood.variance.value

    def get_N(self):
        return 1

    def calc_final_lh(self, X, y):

        self.model.X = X
        self.model.Y = y

        try:
            lh = self.model.compute_log_likelihood()
        except InvalidArgumentError:
            lh = self.lh_logger[-1]

        return lh

    def get_lh_history(self):
        return deepcopy(self.lh_logger)

    def get_parameters_history(self):
        return deepcopy(self.parameters_logger)
