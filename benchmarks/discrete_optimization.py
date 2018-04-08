import numpy as np
import sys
from utilities.utils import read_multi_fidelity_file, read_fidelity_file
from bayesian_optimizer.gp_bucb import BO_Batch


class Sihouxfalls(object):

    def __init__(self, parameters):
        if parameters['BO_Mode'] == 'MES_Fidelity':
            target_distribution, Consts, parameters = \
                read_multi_fidelity_file(parameters['Discrete_Set'],
                                         parameters,
                                         parameters['scenario_dir'])
        else:
            target_distribution, Consts = read_fidelity_file(
                parameters['Discrete_Set'], parameters['Target_Mode'],
                parameters['scenario_dir'])
        parameters['Dimension'] = 3
        parameters['Discrete_Set'] = None
        batch_optimizer = BO_Batch(parameters)
        self.errors = batch_optimizer.calc_error(target_distribution)
        self.Consts = Consts

        best = np.argmax(self.errors)
        print self.Consts[best, :]

    def get_discrete_set(self):
        return self.Consts

    def objective_function(self, X, fidelity):

        y = []
        for row in range(X.shape[0]):
            index = np.where(np.all(self.Consts == X[row, :], axis=1))[0][0]
            y.append(self.errors[index, 0])

        y = np.reshape(np.asarray(y), (-1, 1))

        return y

    def get_meta_information(self):
        return {'name': 'Sihouxfalls',
                'num_function_evals': 200,
                'optima': ([[-0.562, 0.0, -0.124]]),
                'bounds': [[-0.75, 0.0], [-0.75, 0.0], [-0.75, 0.0]],
                'f_opt': 0.0}
