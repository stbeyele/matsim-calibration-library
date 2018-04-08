import numpy as np
import sys


class Hartmann3(object):

    def objective_function(self, x, fidelity):

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        P = 0.0001 * np.array([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1090, 8732, 5547],
                               [381, 5743, 8828]])
        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum = internal_sum + A[i, j] * (x[:, j] - P[i, j])**2

            y = y + alpha[i] * np.exp(-internal_sum)

        y = np.reshape(y, (-1, 1))
        return y

    def get_meta_information(self):
        return {'name': 'Hartmann 3D',
                'num_function_evals': 200,
                'optima': ([[0.114614, 0.555649, 0.852547]]),
                'bounds': [[0, 1], [0, 1], [0, 1]],
                'f_opt': -3.8627795317627736}
