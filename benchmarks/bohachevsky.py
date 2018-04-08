import numpy as np
import sys


class Bohachevsky(object):

    def objective_function(self, x, fidelity):
        y = 0.7 + x[:, 0] ** 2 + 2.0 * x[:, 1] ** 2
        y -= 0.3 * np.cos(3.0 * np.pi * x[:, 0])
        y -= 0.4 * np.cos(4.0 * np.pi * x[:, 1])
        y = np.reshape(y, (-1, 1)) * -1.0
        return y

    def get_meta_information(self):
        return {'name': 'Bohachevsky',
                'num_function_evals': 200,
                'optima': ([[0, 0]]),
                'bounds': [[-100, 100], [-100, 100]],
                'f_opt': 0.0}
