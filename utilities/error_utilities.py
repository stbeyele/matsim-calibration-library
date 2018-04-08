from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import entropy, wasserstein_distance


class Error_Calculator(object):

    def __init__(self, target_distribution, error_mode):

        if error_mode == 'L2':
            self.error_calculator = L2_Error(target_distribution,)
        elif error_mode == 'JSD':
            self.error_calculator = JSD_Distance(target_distribution,)
        elif error_mode == 'Wasserstein':
            self.error_calculator = Wasserstein_Distance(target_distribution,)
        elif error_mode == 'Hellinger':
            self.error_calculator = Hellinger_Distance(target_distribution,)
        else:
            raise ValueError(error_mode + ' is not a valid error mode')

    def calc_error(self, matsim_distribution):
        """ Calculate l2 error for target mode"""

        return self.error_calculator.calc_error(matsim_distribution)

class Abstract_Error(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calc_error(self, share):
        """test"""
        return

class L2_Error(object):

    def __init__(self, target):

        self.target = target

    def calc_error(self, share):
        """Calculate l2 error for mode share distribution"""

        shape = share.shape
        rows = shape[0]
        columns = shape[1]

        l2_errors = np.zeros((1, columns))
        for idx in range(rows):
            l2_errors = l2_errors + np.power(
                share[idx, :] - self.target[idx], 2)
        l2_errors = np.reshape(l2_errors, (-1, 1))

        return np.sqrt(l2_errors) * -1


Abstract_Error.register(L2_Error)


class Hellinger_Distance(object):

    def __init__(self, target):

        self.target = target

    def calc_error(self, share):
        """Calculate heillinger distance"""

        shape = share.shape
        rows = shape[0]
        columns = shape[1]

        heillinger_dist = np.zeros((columns))
        for idx in range(rows):
            heillinger_dist = heillinger_dist + np.power(
                np.sqrt(share[idx, :]) - np.sqrt(self.target[idx]), 2)
        heillinger_dist = np.reshape(heillinger_dist, (-1, 1))

        return np.sqrt(heillinger_dist) * -1 / np.sqrt(2)


Abstract_Error.register(Hellinger_Distance)


class JSD_Distance(object):

    def __init__(self, target):

        self.target = target

    def calc_jsd(self, share):
        """Calculate jsd distance"""
        columns = share.shape[1]

        jsd_distances = np.zeros((columns, 1))

        for column in range(columns):
            jsd_distances[column, 0] =\
                self.calc_jsd_distance(share[:, column], self.target)

        return jsd_distances * -1

    def calc_jsd_distance(self, p, q):
        """similarity between 0 and 1"""

        p = p / np.linalg.norm(p, ord=1)
        q = q / np.linalg.norm(q, ord=1)
        m = 0.5 * (p + q)

        return np.sqrt(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


Abstract_Error.register(JSD_Distance)


class Wasserstein_Distance(object):

    def __init__(self, target):

        self.target = target

    def calc_wasserstein(self, share):
        """Calculate Wasserstein distance"""

        columns = share.shape[1]

        wasserstein_distances = np.zeros((columns, 1))

        for column in range(columns):
            wasserstein_distances[column, 0] =\
                wasserstein_distance(share[:, column], self.target)

        return wasserstein_distances * -1


Abstract_Error.register(Wasserstein_Distance)
