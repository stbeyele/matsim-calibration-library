import numpy as np
import sys


class Levy(object):

    def objective_function(self, x, fidelity):
        z = 1.0 + ((x[:, 0] - 1.0) / 4.0)
        s = np.power((np.sin(np.pi * z)), 2)
        y = (s + ((z - 1.0)**2) * (1.0 +
                                   np.power((np.sin(2.0 * np.pi * z)), 2)))

        y = np.reshape(y, (-1, 1)) * -1.0
        return y

    def get_meta_information(self):
        return {'name': 'Levy',
                'num_function_evals': 200,
                'optima': ([[1.0]]),
                'bounds': [[-15, 10]],
                'f_opt': 0.0}
