from abc import ABCMeta, abstractmethod
from bayesian_optimizer.gp_bucb import GP_BUCB
from bayesian_optimizer.b_mes import B_MES
from bayesian_optimizer.mfb_mes import MFB_MES
from bayesian_optimizer.gp_bucb_dim import GP_BUCB_Dimension
from bayesian_optimizer.random import Random_Batch

class Bayesian_Optimizer(object):

    def __init__(self, parameters):

        algorithm = parameters['BO_Mode']
        self.optimizer = None

        if algorithm == 'GP-BUCB':
            self.optimizer = GP_BUCB(parameters)
        elif algorithm == 'B-MES':
            self.optimizer = B_MES(parameters)
        elif algorithm == 'GP-BUCB-Dimension':
            self.optimizer = GP_BUCB_Dimension(parameters)
        elif algorithm == 'MFB_MES':
            self.optimizer = MFB_MES(parameters)
        elif algorithm == 'Random':
            self.optimizer = Random_Batch(parameters)
        else:
            raise ValueError(parameters['BO_Mode'] +
                             ' is not a valid bo algorithm')

    def init_bo(self, batch_size, X_train, y_train):

        self.optimizer.init_bo(batch_size, X_train, y_train)

    def optimize_function(self):

        self.optimizer.optimize_function()

    def update_model(self, X_new, y_new):

        self.optimizer.update_model(X_new, y_new)

    def get_bo_history(self):

        self.optimizer.get_bo_history()

    def set_batch_size(self, batch_size):

        self.optimizer.set_batch_size(batch_size)

    def optimize_model(self, Consts, errors, kernel_list):

        return self.optimizer.optimize_model(Consts, errors, kernel_list)

    def get_new_sample_points(self):

        return self.optimizer.get_new_sample_points()


class Abstact_Optimizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_bo(self, batch_size, X_train, y_train):
        """test"""
        return

    @abstractmethod
    def optimize_function(self):
        """Test"""
        return

    @abstractmethod
    def update_model(self, X_new, y_new):
        """Test"""
        return

    @abstractmethod
    def get_bo_history(self):
        """Test"""
        return

    @abstractmethod
    def set_batch_size(self, batch_size):
        """Test"""
        return

    @abstractmethod
    def optimize_model(self, Consts, errors, kernel_list):
        """Test"""
        return

    @abstractmethod
    def get_new_sample_points(self):

        return


Abstact_Optimizer.register(GP_BUCB)
Abstact_Optimizer.register(B_MES)
Abstact_Optimizer.register(MFB_MES)
Abstact_Optimizer.register(GP_BUCB_Dimension)

