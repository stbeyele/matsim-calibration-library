from simulator.matsim.batch_run import Batch_Run
from simulator.matsim.analyzer import Result_Analyser

class MATSim(object):

    def __init__(self, simulator_paths, parameters):

        self.analyser = Result_Analyser(parameters)
        self.matsim = Batch_Run(simulator_paths, parameters)

    def get_errors(self, batch_size, data_paths):

        return self.analyser.get_errors(batch_size, data_paths)

    def get_runtimes(self, data_paths):

        return self.analyser.get_runtimes(data_paths)

    def update_batch_size(self, batch_size):

        self.matsim.update_batch_size(batch_size)

    def run_check_batches(self, parameters, offset, run):

        return self.matsim.run_check_batches(parameters, offset, run)

    def get_bins(self):

        return self.analyser.get_bins()

