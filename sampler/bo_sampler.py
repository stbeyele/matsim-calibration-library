from abc import ABCMeta, abstractmethod
from abc_base import PluginBasess
from sampler.point_sampler import Point_Sampler
from sampler.slice_sampler import Slice_Sampler


class BO_Sampler(object):

    def __init__(self, parameters, model, set_parameters):

        sampler = parameters['Sampler']

        if sampler == 'PointSampler':
            self.bo_sampler = Point_Sampler(parameters, model, set_parameters)
        elif sampler == 'SliceSampler':
            self.bo_sampler = Slice_Sampler(parameters, model, set_parameters)
        else:
            raise ValueError(parameters['Sampler'] +
                             ' is not a valid sampler')

    def predict(self, X, y, X_pred):

        self.bo_sampler.predict(X, y, X_pred)

    def predict_sample(self, X, y, X_pred, i):

        self.bo_sampler.predict_sample(X, y, X_pred)

    def predict_samples(self, X, y, X_pred):

        self.bo_sampler.predict_samples(X, y, X_pred)

    def optimize(self, X, y, X_pred):

        self.bo_sampler.optimize(X, y, X_pred)


class Abstract_Sampler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, X, y, X_pred):
        """test"""
        return

    @abstractmethod
    def predict_sample(self, X, y, X_pred, idx):
        """test"""
        return

    @abstractmethod
    def predict_samples(self, X, y, X_pred):
        """test"""
        return

    @abstractmethod
    def optimize(self, X, y, X_pred):
        """test"""
        return


Abstract_Sampler.register(Point_Sampler)
Abstract_Sampler.register(Slice_Sampler)

