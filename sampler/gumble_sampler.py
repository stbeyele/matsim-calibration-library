import sys
from copy import copy, deepcopy
import numpy as np
from scipy.stats import norm
from utilities.runner_utilities import convert_fidelity_matrix

class Gumble_Sampler(object):

    def __init__(self, parameters, opt_parameters, dim):
        self.parameters = deepcopy(parameters)
        self.opt_parameters = deepcopy(opt_parameters)
        self.m = None
        self.X_var = None
        self.y_var = None
        self.X = None
        self.y = None
        self.dim = dim
        self.local_fidelity = False

    def get_maxes_start_values(self, model, X, y, X_var, y_var, fidelity):

        np.random.seed(self.parameters['Seed'])
        self.X = X
        self.y = y
        self.X_var = X_var
        self.y_var = y_var
        self.m = model

        xmin = np.asarray(self.opt_parameters['Lower_Bounds'])
        xmax = np.asarray(self.opt_parameters['Upper_Bounds'])
        K = self.parameters['K']
        gridsize = self.parameters['GridSize']
        mgridsize = self.parameters['MGridSize']
        N = self.m.get_N()
        maxes = np.zeros((N, K))
        yvals = 0.0

        for n in range(N):

            mu, std_dev, Xgrid, w = self.get_mean_variance(xmin, xmax,
                                                           gridsize, n,
                                                           fidelity)

            maxes[n, :] = self.sample_maximums(mu, std_dev, mgridsize, K,
                                               Xgrid.shape[0], n)

            yvals = self.get_start_value(mu, std_dev, maxes[n, :], Xgrid, K,
                                         yvals, w)

        yvals = yvals / (float(K) * float(N))
        maxIdx = np.argmax(yvals)
        start = Xgrid[maxIdx, :]

        for j in range(start.shape[0]):
            if start[j] > self.opt_parameters['Upper_Bounds'][j]:
                start[j] = self.opt_parameters['Upper_Bounds'][j]
            if start[j] < self.opt_parameters['Lower_Bounds'][j]:
                start[j] = self.opt_parameters['Lower_Bounds'][j]

        return start, maxes

    def get_variance(self, x, idx):
        """Get Variance of GP"""

        _, var, w = self.m.predict_sample(self.X_var, self.y_var, x, idx)

        return var, w

    def get_mean(self, x, idx):
        """Get mean of GP"""

        y, _, w = self.m.predict_sample(self.X, self.y, x, idx)

        return y, w

    def get_mean_variance(self, xmin, xmax, gridsize, idx, fidelity):
        """Calculate mean and variance for grid values"""

        if self.local_fidelity:
            fidelity = deepcopy(fidelity)
        else:
            fidelity = self.parameters['Fidelities'] - 1

        Xgrid = (np.matlib.repmat(xmin, gridsize, 1) +
                 np.matlib.repmat((xmax - xmin), gridsize, 1) *
                 np.random.rand(gridsize, self.dim))
        if self.parameters['Fidelities'] > 0:
            X_fidelity = convert_fidelity_matrix(
                self.parameters['Fidelities'],
                fidelity,
                Xgrid.shape[0],
                np.ones((self.parameters['Fidelities'])))  # * fidelity)
        # X_fidelity = convert_fidelity_matrix(self.parameters['Fidelities'],
        #                                      self.fidelity_choice,
        #                                      Xgrid.shape[0]) #old approach
            Xgrid = np.hstack((Xgrid, X_fidelity))
        Xgrid = np.vstack((Xgrid, self.X))
        mu, w = self.get_mean(Xgrid, idx)
        mu = np.reshape(mu, (-1, 1))

        std_dev, w = self.get_variance(Xgrid, idx)
        std_dev = np.reshape(np.sqrt(std_dev), (-1, 1))

        return mu, std_dev, Xgrid[:, 0:self.dim], w

    def sample_maximums(self, mu, std_dev, mgridsize, K, sx, idx):
        """Sample K maximums using Gumble sampling"""

        left = np.max(self.y)
        if self.probf(left, mu, std_dev) < 0.25:

            right = np.max(mu + 5 * std_dev)
            while self.probf(right, mu, std_dev) < 0.75:
                right = right + right - left

            mgrid = np.linspace(left, right, mgridsize)
            prob = np.prod(
                norm.cdf((np.matlib.repmat(mgrid, sx, 1) -
                          np.matlib.repmat(mu, 1, mgridsize)) /
                         np.matlib.repmat(std_dev, 1, mgridsize)), 0)
            med = self.find_between(
                0.5, self.probf, prob, mgrid, 0.01, mu, std_dev)
            q1 = self.find_between(
                0.25, self.probf, prob, mgrid, 0.01, mu, std_dev)
            q2 = self.find_between(
                0.75, self.probf, prob, mgrid, 0.01, mu, std_dev)

            beta = (q1 - q2) / (np.log(np.log(4.0 / 3.0)) -
                                np.log(np.log(4.0)))
            alpha = med + beta * np.log(np.log(2.0))
            maxes = -np.log(-np.log(np.random.rand(K))) * beta + alpha

            indices = np.where(maxes < (left + 5 * np.sqrt(
                self.m.get_lh_variance(idx))))
            maxes[indices] = left + 5 * np.sqrt(
                self.m.get_lh_variance(idx))

        else:
            maxes = np.ones(K) * (left + 5 *
                                  np.sqrt(self.m.get_lh_variance(idx)))

        return maxes

    def get_start_value(self, mu, std_dev, maxes, Xgrid, K, yvals, w):
        """Get start value for efficient optimization"""

        sx = Xgrid.shape[0]
        gamma = ((np.matlib.repmat(maxes, sx, 1) -
                  np.matlib.repmat(mu, 1, K)) /
                 np.matlib.repmat(std_dev, 1, K))
        pdfgamma = norm.pdf(gamma)
        cdfgamma = norm.cdf(gamma)

        yvals = yvals + w * np.sum(gamma * pdfgamma / (2 * cdfgamma) -
                                   np.log(cdfgamma), 1)

        return yvals

    def probf(self, m0, mu, std):
        """Caluclate probability function"""

        return np.prod(norm.cdf((m0 - mu) / std))


    def find_between(self, val, func, funcvals, mgrid, thres, mu, std_dev):
        """Find best value for x given a grid of y values"""

        #  index = np.argmin(abs(funcvals - val))
        temp = abs(funcvals - val)
        temp = temp[::-1]
        index = len(temp) - np.argmin(temp) - 1
        if index + 1 >= mgrid.shape[0] and funcvals[index] <= val:
            return mgrid[index]


        if abs(funcvals[index] - val) < thres:
            res = mgrid[index]
            return res

        if funcvals[index] > val:
            left = mgrid[index - 1]
            right = mgrid[index]
        else:
            left = mgrid[index]
            right = mgrid[index + 1]

        mid = (left + right) / 2
        midval = func(mid, mu, std_dev)

        while abs(midval - val) > thres:
            if midval > val:
                right = mid
            else:
                left = mid
            mid = (left + right) / 2
            midval = func(mid, mu, std_dev)

        return mid
