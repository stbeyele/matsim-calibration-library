import numpy as np
import sys
from copy import copy


class Forrester(object):

    def objective_function(self, x, fidelity):

        if fidelity:
            fidelity = copy(x[:, 1])
            fidelity[np.where(fidelity == 0)] = 0.1

        else:
            fidelity = np.ones(x.shape[0])

        x = x[:, 0]
        y1 = np.power(6.0 * x - 2.0, 2) * np.sin(12.0 * x - 4.0)

        # best least-squared fit with cubic polynomial
        y2 = 131.09227753 * (x**3) - 164.50286816 * (x**2) +\
            50.7228373 * x - 2.84345244
        y = fidelity * y1 + (1 - fidelity) * y2
        y = np.reshape(y, (-1, 1)) * -1.0
        #  'cost': fidelity**2
        return y

    def get_meta_information(self):
        return {'name': 'Branin',
                'num_function_evals': 20,
                'optima': ([[0.75724875]]),
                'bounds': [[0, 1]],
                'f_opt': -6.02074}
