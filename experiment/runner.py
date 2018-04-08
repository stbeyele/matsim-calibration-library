"""Runs the different experiments"""
import sys
import os
import time
from copy import copy
import csv
import timeit
import pickle
import yaml
import numpy as np
from simulator.abstract_simulator import Simulator
from utilities.runner_utilities import \
    delete_dict_entries, delete_histroy, read_fidelity_file, \
    update_optimization_results, calc_error_parameter, \
    mes_fidelity_helper, reduce_dim_helper, create_parameters, \
    read_multi_fidelity_file, create_csv_target, create_csv_runtime, \
    save_pickle_kernel_parameters, \
    convert_fidelity_matrix, run_fidelity_iterations, read_data_file
from utilities.io_utilities import check_create_dir
from utilities.error_utilities import Error_Calculator
from sampler.data_sampler import Parameter_Sampling
from bayesian_optimizer.optimizer import Bayesian_Optimizer
from bayesian_optimizer.analyzer import BO_Analyser





class Run_Experiments(object):
    """ Run different experiments """

    def __init__(self, paths):

        self.paths = copy(paths)

    def fidelities(self, parameters, output_dir):
        """Runs the fidelity experiment"""

        self.paths['output_dir'] = output_dir
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = 'fidelities_' + timestr
        parameters = copy(parameters)
        batch_size = parameters['Batch_Size']
        output_file = copy(parameters)
        sim_batch = Simulator(self.paths, parameters)

        sampler = Parameter_Sampling(parameters)
        sampled_parameters, experiment_length = sampler.get_samples()

        parameters.update(sampled_parameters)
        output_file.update(sampled_parameters)

        for parameter, value in parameters.iteritems():
            if not isinstance(value, list):
                parameters[parameter] = [value] * experiment_length

        if experiment_length >= batch_size:
            batch_runs = int(experiment_length / batch_size)
            remaining_runs = experiment_length % batch_size
            additional_run = int(remaining_runs > 0)
        else:
            batch_size = experiment_length
            batch_runs = 1
            remaining_runs = 0
            additional_run = 0

        offset = 0
        analysis_dirs = []

        for run in range(batch_runs + additional_run):

            if run == batch_runs:
                batch_size = remaining_runs
                sim_batch.update_batch_size(batch_size)

            output_dirs_batch = sim_batch.run_check_batches(parameters, offset,
                                                            0)

            analysis_dirs = analysis_dirs + delete_histroy(
                self.paths, output_dirs_batch,
                self.paths['delete_output_history'], True,
                filename)

            offset = offset + batch_size

        fidelities_folder = self.paths['log_dir'] + 'fidelities/'
        check_create_dir(fidelities_folder)
        with open(fidelities_folder + filename + '.txt', 'w') as f:
            yaml.dump(output_file, f, default_flow_style=False)

        pickle_name = os.path.join(self.paths['log_dir'], 'output_files',
                                   filename, 'parameters.pickle')
        output_file['paths'] = analysis_dirs
        output_file['filename'] = filename
        pickle.dump(output_file, open(pickle_name, "wb"))

        self.analyze_output(output_file, output_dir)

        return filename

    def analyze_output(self, parameters, output_dir):
        """Analyzes MATSim output files"""

        target_mode = parameters['Target_Mode']
        if 'Directory' in parameters:

            pickle_file = os.path.join(parameters['Directory'],
                                       'parameters.pickle')
            parameters = pickle.load(open(pickle_file, "rb"))
            parameters['Target_Mode'] = target_mode

        filename = parameters['filename']
        fidelities_name = os.path.join(self.paths['log_dir'], 'fidelities',
                                       filename)
        paths = parameters['paths']
        del parameters['filename']
        del parameters['paths']

        csv_parameters = []
        for parameter, value in parameters.iteritems():
            if not isinstance(value, list):
                value = [value]
            csv_parameters.append([parameter] + value)

        analyzer = Simulator(self.paths, parameters)
        output_csv_target = create_csv_target(analyzer, paths,
                                              copy(csv_parameters))
        with open(fidelities_name + '_' + target_mode + '.csv', 'wb') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(output_csv_target)

        output_csv_runtime = create_csv_runtime(analyzer,
                                                paths, copy(csv_parameters))
        with open(fidelities_name + '_runtimes.csv', 'wb') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(output_csv_runtime)

        return 'analysis_' + filename

    def optimize_bayesian(self, parameters, output_dir):
        """Train GP model"""

        kernel_list = parameters['Kernels']
        output = {}
        output.update(copy(parameters))

        distribution, Consts, parameters = read_data_file(self.paths,
                                                          parameters)

        optimizer = Bayesian_Optimizer(parameters)

        error_calculator = Error_Calculator(parameters['Distribution_Target'],
                                            parameters['Error_Mode'])

        errors = error_calculator.calc_error(distribution)

        start = timeit.default_timer()
        model, result = optimizer.optimize_model(
            Consts, errors, kernel_list)
        stop = timeit.default_timer()
        runtime = stop - start

        output = update_optimization_results(output, model, result, runtime)

        optimization_folder = self.paths['log_dir'] + 'optimization/'
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = 'optimization_' + timestr
        check_create_dir(optimization_folder)
        with open(optimization_folder + filename + '.txt', 'w') as f:
            yaml.dump(output, f, default_flow_style=False)

        save_pickle_kernel_parameters(output, optimization_folder,
                                      filename)

        return filename

    def bayesian_batch(self, parameters, output_dir):
        """Run Bayesian Optimization"""

        self.paths['output_dir'] = output_dir
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = 'bo_batch_' + timestr
        output_file = copy(parameters)
        parameters = copy(parameters)
        batch_size = parameters['Batch_Size']
        runs = parameters['Runs']
        mode = parameters['BO_Mode']
        ground_truth = np.asarray(parameters['Ground_Truth'])
        parameters['Dimension'] = len(parameters['Lower_Bounds'])

        if 'Parameters_File' in parameters:
            kernel_parameters = pickle.load(
                open(parameters['Parameters_File'], "rb"))
            parameters.update(kernel_parameters)

        error_calculator = Error_Calculator(parameters['Distribution_Target'],
                                            parameters['Error_Mode'])

        if 'Input_Data' in parameters:
            distribution, Consts, parameters = read_data_file(self.paths,
                                                              parameters)
            errors = error_calculator.calc_error(distribution)

        else:
            Consts = None
            errors = None

        batch_optimizer = Bayesian_Optimizer(parameters)

        batch_optimizer.init_bo(batch_size, Consts, errors)

        simulator = Simulator(self.paths, parameters)

        offset = 0
        run_times = {}

        for parameter, value in parameters.iteritems():
            parameters[parameter] = [value] * batch_size * 2

        distribution_history = np.zeros((len(simulator.get_bins()), 1))

        for run in range(runs):
            X = batch_optimizer.get_new_sample_points()

            if mode == 'MFB_MES':
                parameters = mes_fidelity_helper(X, parameters, output_file)
            else:
                parameters = create_parameters(X, parameters)

            output_dirs = simulator.run_check_batches(
                parameters, offset, run)
            print '----------------' + str(run) + '------------------'
            print X

            distribution = simulator.get_errors(batch_size, output_dirs)
            distribution_history = np.hstack((
                distribution_history, distribution))
            errors = error_calculator.calc_error(distribution)
            batch_optimizer.update_model(X, errors)

            run_times.update(simulator.get_runtimes(output_dirs))

            delete_histroy(self.paths, output_dirs,
                           self.paths['delete_output_history'],
                           self.paths['save_files'], filename)

        X, y, y_full_fidelity = batch_optimizer.get_bo_history()
        print X
        print y
        print y_full_fidelity

        output_file['L2_error'], output_file['Best_Parameters'] = \
            calc_error_parameter(
                X, y_full_fidelity, ground_truth, parameters['Dimension'][0])

        bo_batch_folder = self.paths['log_dir'] + 'bo_batch/'
        check_create_dir(bo_batch_folder)

        plotter = BO_Analyser(batch_size, filename, bo_batch_folder, False)
        plotter.plot_bo(y, 0.0)
        plotter.save_figure()

        pickle_file = {
            'X': X, 'y': y, 'y_full_fidelity': y_full_fidelity,
            'target_distribution': distribution_history[:, 1:],
            'batch_size': batch_size, 'runs': runs}
        with open(bo_batch_folder + filename + '.pickle', 'wb') as handle:
            pickle.dump(pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(bo_batch_folder + filename + '.txt', 'w') as f:
            yaml.dump(output_file, f, default_flow_style=False)
        with open(bo_batch_folder + filename + '_runtimes.txt', 'w') as f:
            yaml.dump(run_times, f, default_flow_style=False)

        return filename
