import numpy as np
import sys


class GoldsteinPrice(object):

    def objective_function(self, x, fidelity):
        y = (1 + (x[:, 0] + x[:, 1] + 1)**2 * (
            19 - 14 * x[:, 0] + 3 * x[:, 0]**2 - 14 * x[:, 1] + 6 * x[:, 0] *
            x[:, 1] + 3 * x[:, 1] ** 2)) *\
            (30 + (2 * x[:, 0] - 3 * x[:, 1])**2 * (
                18 - 32 * x[:, 0] + 12 * x[:, 0]**2 + 48 * x[:, 1] - 36 *
                x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2))

        y = np.reshape(y, (-1, 1)) * -1.0
        return y

    def get_meta_information(self):
        return {'name': 'GoldsteinPrice',
                'num_function_evals': 200,
                'optima': ([[0.0, -1.0]]),
                'bounds': [[-2, 2], [-2, 2]],
                'f_opt': 3}
