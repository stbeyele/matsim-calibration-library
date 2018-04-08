import numpy as np
import sys


class Hartmann6(object):

    def objective_function(self, x, fidelity):

        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])

        y = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum = internal_sum + A[i, j] * (x[:, j] - P[i, j])**2
            y = y + alpha[i] * np.exp(-internal_sum)

        y = np.reshape(y, (-1, 1))
        return y

    def get_meta_information(self):
        return {'name': 'Hartmann 3D',
                'num_function_evals': 200,
                'optima': ([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652,
                             0.6573]]),
                'bounds': [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                'f_opt': -3.322368011391339}
