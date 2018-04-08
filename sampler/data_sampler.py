"""This module samples the points for training the hyperparameters"""
import itertools
import sys
import numpy as np
from pyDOE import lhs
from abc import ABCMeta, abstractmethod


class Parameter_Sampling(object):
    """ Do sampling according to given sampling mode"""

    def __init__(self, parameters):

        if parameters['Sample_Mode'] == 'Grid':
            self.parameter_sampler = Grid_Sampling(parameters)

        elif parameters['Sample_Mode'] == 'LHS':
            self.parameter_sampler = LHS_Sampling(parameters)

        else:
            raise ValueError(parameters['Sample_Mode'] +
                             ' is not a valid sample mode')

    def get_samples(self):

        parameters, experiment_length = self.parameter_sampler.get_samples()

        return parameters, experiment_length

class Abstract_Sampling(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_samples(self):
        """test"""
        return

class Grid_Sampling(object):
    """ Grid Sampling """

    def __init__(self, parameters):
        self.parameters = parameters

    def get_samples(self):

        experiment_length = []
        combination_list = []
        combination_parameters = []
        sampled_parameters = {}

        for parameter in self.parameters['Parameters']:

            new_value = np.linspace(self.parameters[parameter][0],
                                    self.parameters[parameter][1],
                                    self.parameters[parameter][2])
            experiment_length.append(len(new_value))
            combination_list.append(new_value)
            sampled_parameters[parameter] = []
            combination_parameters.append(parameter)

        #  add all possible combinations to parameters
        for combination in itertools.product(*combination_list):
            for idx, val in enumerate(combination):
                sampled_parameters[combination_parameters[idx]].append(
                    val.tolist())

        experiment_length = int(np.prod(experiment_length))

        return sampled_parameters, experiment_length


Abstract_Sampling.register(Grid_Sampling)


class LHS_Sampling(object):
    """ Do Latin Hyper-Cube Sampling"""

    def __init__(self, parameters):

        self.parameters = parameters

    def get_samples(self):

        np.random.seed(self.parameters['Seed'])
        sample_number = self.parameters['Sample_Number']
        random_mode = self.parameters['LHS_Mode']
        sampled_parameters = {}

        Constants = []
        lower_bounds = []
        upper_bounds = []

        for parameter in self.parameters['Parameters']:
            Constants.append(parameter)
            lower_bounds.append(self.parameters[parameter][0])
            upper_bounds.append(self.parameters[parameter][1])

        dim = len(Constants)

        if random_mode == 'None':
            random_mode = None

        samples = lhs(dim, sample_number, random_mode)

        for idx in range(dim):
            sampled_parameters[Constants[idx]] = (
                samples[:, idx] * (upper_bounds[idx] - lower_bounds[idx]) +
                lower_bounds[idx])
            sampled_parameters[Constants[idx]] =\
                sampled_parameters[Constants[idx]].tolist()

        return sampled_parameters, sample_number


Abstract_Sampling.register(LHS_Sampling)

