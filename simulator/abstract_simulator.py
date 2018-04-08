from abc import ABCMeta, abstractmethod
from simulator.matsim.matsim import MATSim


class Simulator(object):

    def __init__(self, simulator_paths, parameters):

        simulator = simulator_paths['Simulator']

        if simulator == 'MATSim':
            self.simulator = MATSim(simulator_paths, parameters)
        else:
            raise ValueError(simulator +
                             ' is not a valid simulator')

    def get_errors(self, batch_size, data_paths):

        return self.simulator.get_errors(batch_size, data_paths)

    def get_runtimes(self, data_paths):

        return self.simulator.get_runtimes(data_paths)

    def update_batch_size(self, batch_size):

        self.simulator.update_batch_size(batch_size)

    def run_check_batches(self, parameters, offset, run):

        return self.simulator.run_check_batches(parameters, offset, run)

    def get_bins(self):

        return self.simulator.get_bins()


class Abstract_Simulator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_errors(self, batch_size, paths):
        """test"""
        return

    @abstractmethod
    def get_runtimes(self, paths):
        """test"""
        return

    @abstractmethod
    def update_batch_size(self, batch_size):
        """test"""
        return

    @abstractmethod
    def run_check_batches(self, config, parameters, offset, run, X):
        """test"""
        return

    @abstractmethod
    def get_bins(self):

        return

Abstract_Simulator.register(MATSim)
