import numpy as np
import sys


class Rosenbrock(object):

    def objective_function(self, x, fidelity):
        y = 0.0
        d = 2
        for i in range(d - 1):
            y += 100.0 * (x[:, i + 1] - x[:, i]**2)**2
            y += (x[:, i] - 1.0)**2

        y = np.reshape(y, (-1, 1)) * -1.0
        return y

    def get_meta_information(self):
        return {'name': 'Rosenbrock',
                'num_function_evals': 200,
                'optima': ([[1, 1]]),
                'bounds': [[-5, 10], [-5, 10]],
                'f_opt': 0.0}
