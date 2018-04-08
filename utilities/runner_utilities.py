"""Set of utility function for data loading and manipulation"""
import os
import sys
import shutil
import csv
import pickle
from copy import copy
import numpy as np
from simulator.abstract_simulator import Simulator


def delete_dict_entries(dictionary, entries):
    """ Delete a list of keys in dictionary"""

    for key in entries:
        if key in dictionary:
            del dictionary[key]

    return dictionary
    # parameters = {key: parameters[key] for key in parameters if key not in del_parameter}


def delete_histroy(config, output_paths, delete_history, save_files,
                   folder_name):
    """ Delete temporary and not needed MATSim output files"""

    # if os.path.exists(config['temp_dir']):
    #     shutil.rmtree(config['temp_dir'])

    for idx, output_path in enumerate(output_paths):

        old_path = output_paths[idx]['output_dir']

        if save_files is True:

            target_folder = config['log_dir'] + 'output_files/' + \
                folder_name + '/' + str(output_path['run_counter']) + '/'
            check_create_dir(target_folder)

            event_file = output_path['events_file']
            shutil.copyfile(output_path['output_dir'] + event_file,
                            target_folder + event_file)

            trip_file = output_path['tripduration_file']
            trip_file_path, _ = os.path.split(target_folder + trip_file)
            check_create_dir(trip_file_path)
            shutil.copyfile(output_path['output_dir'] + trip_file,
                            target_folder + trip_file)

            stopwatch_file = output_path['stopwatch_file']
            shutil.copyfile(output_path['output_dir'] + stopwatch_file,
                            target_folder + stopwatch_file)

            output_paths[idx]['output_dir'] = target_folder

        if delete_history is True:

            if os.path.exists(old_path):
                shutil.rmtree(old_path)

    return output_paths

def read_data_file(paths, parameters):

    mode = parameters['BO_Mode']
    data_path = parameters['Input_Data']
    analyser = Simulator(paths, parameters)

    if mode == 'MFB-MES':
        distribution, Consts, parameters = \
            read_multi_fidelity_file(analyser, data_path, parameters)
    else:
        distribution, Consts = read_fidelity_file(
            analyser, data_path, parameters['Parameters'])
    parameters['Dimension'] = Consts.shape[1]

    return distribution, Consts, parameters


def read_fidelity_file(analyser, data_path, simulator_parameters):
    """Read file from fidelity experiments and return distr. and param."""

    input_data = {}
    with open(data_path, 'rb') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            name = row[0]
            data = row
            data.pop(0)
            input_data[name] = data

    bins = analyser.get_bins()
    distribution_bins = []
    for distribution_bin in bins:
        distribution_bins.append(distribution_bin + '_Share')

    distribution = get_array(input_data, distribution_bins)

    parameters = np.transpose(get_array(input_data, simulator_parameters))

    return distribution, parameters


def get_array(input_data, modes):
    """Converts dict to array"""

    shares = []
    for mode in modes:
        shares.append([float(i) for i in input_data[mode]])

    shares_array = np.asarray(shares.pop(0))
    for share in shares:
        shares_array = np.vstack((shares_array, share))

    return shares_array


def update_optimization_results(output, model, result, runtime):
    """updates dict with training results"""

    data_path = output['Input_Data']

    result['Optimized_Model'] = [model]
    result['Optimized_Likelihood'] = [float(
        result['Optimized_Likelihood'])]

    if isinstance(data_path, list):
        result['Dataset'] = []
        for data in data_path:
            result['Dataset'].append(os.path.basename(data))
    else:
        result['Dataset'] = [os.path.basename(data_path)]

    result['Runtime'] = runtime

    if output['Kernel_Transform'] is not False:
        result['Kernel_Transform'] = str(output['Kernel_Transform'][0]) + \
            '_' + str(output['Kernel_Transform'][1])
    else:
        result['Kernel_Transform'] = output['Kernel_Transform']

    for key, value in result.iteritems():
        output[key] = value

    return output


def calc_error_parameter(X, y, target, dimension): #change if more parameters
    """Calculate error of parameters"""

    pos_max = np.argmax(y)
    best_parameters = X[pos_max, 0:dimension]
    best_parameters = np.reshape(best_parameters, (-1, 1))

    l2_errors = (
        np.power(best_parameters[0, :] - target[0], 2) +
        np.power(best_parameters[1, :] - target[1], 2) +
        np.power(best_parameters[2, :] - target[2], 2))

    return l2_errors.tolist(), best_parameters.tolist()


def mes_fidelity_helper(X, parameters, output_file):
    """Sets opulation and iteration parameters for a given fidelity"""

    pop_fractions = output_file['Population_Fraction']
    iterations = output_file['Iterations']

    chosen_fidelity = X[:, parameters['Dimension'][0]:]
    chosen_fidelity = np.sum(chosen_fidelity, axis=1).tolist()

    parameters['Population_Fraction'] = \
        [pop_fractions[int(idx)] for idx in chosen_fidelity]
    parameters['Iterations'] = \
        [iterations[int(idx)] for idx in chosen_fidelity]

    parameters = create_parameters(X, parameters)

    return parameters


def mes_fabolas_helper(X, parameters, output_file):
    """Sets opulation and iteration parameters for a given fidelity"""

    fidelity_feature = parameters['Fidelity_Feature']
    chosen_fidelity = X[:, -1]

    if fidelity_feature == 'Population':
        parameters['Population_Fraction'] = chosen_fidelity.tolist()
    else:
        parameters['Iterations'] = chosen_fidelity.tolist()

    parameters = create_parameters(X, parameters)

    return parameters


def reduce_dim_helper(X, parameters, encoder):
    """Recovers full dimensionality of parameters"""

    X = encoder.decode(X)
    parameters = create_parameters(X, parameters)

    return parameters


def create_parameters(X, parameters):
    """Sets Constants in parameters"""

    parameters['Car_Constant'] = X[:, 0].tolist()
    parameters['Walk_Constant'] = X[:, 1].tolist()
    parameters['PT_Constant'] = X[:, 2].tolist()

    return parameters


def read_multi_fidelity_file(analyser, data_path, parameters):
    """Read fidelity file from a certain fidelity"""

    fidelity_path = copy(data_path)
    simulator_parameters = parameters['Parameters']

    fidelities = len(data_path)
    factor = np.ones(fidelities)

    parameters['Fidelities'] = fidelities

    target_distribution, Consts = read_fidelity_file(
        analyser, fidelity_path.pop(0), simulator_parameters)

    parameters['Dimension'] = Consts.shape[1]
    Consts = np.hstack(
        (Consts, convert_fidelity_matrix(fidelities, 0, Consts.shape[0],
                                         factor)))

    for j, data in enumerate(fidelity_path):
        target_distribution_fidelity, Consts_fidelity = \
            read_fidelity_file(analyser, data, simulator_parameters)
        target_distribution = np.hstack(
            (target_distribution, target_distribution_fidelity))
        fidelity_matrix = convert_fidelity_matrix(fidelities, j + 1,
                                                  Consts_fidelity.shape[0],
                                                  factor)
        Consts_fidelity = np.hstack((Consts_fidelity,
                                     fidelity_matrix))
        Consts = np.vstack((Consts, Consts_fidelity))

    return target_distribution, Consts, parameters


def convert_fidelity_matrix(fidelities, fidelity, sx, factor):
    """Covnerts a given fidelity level to a matrix"""

    matrix = np.zeros((sx, fidelities - 1))
    for idx in range(0, fidelity):
        matrix[:, idx] = np.ones((sx)) * factor[fidelity]

    return matrix


def create_csv_target(analysis, paths, csv_parameters):
    """Create a csv containing target distribution and parameters"""

    target_distribution = analysis.get_errors(
        5, paths)

    output_csv_target = csv_parameters
    for key, value in target_distribution.iteritems():
        output_csv_target.append([key + '_Share'] + value)

    return output_csv_target


def create_csv_runtime(analysis, paths, csv_parameters):
    """Create a csv conaitning runtimes and parameters"""

    run_times = analysis.get_runtimes(paths)
    runs = max(run_times.keys()) + 1

    output_csv_runtime = csv_parameters
    runtimes = ['Runtimes']
    for run in range(runs):
        runtimes.append(run_times[run])
    output_csv_runtime.append(runtimes)

    return output_csv_runtime


def save_pickle_kernel_parameters(parameters, output_folder, filename):
    """ Save found kernel parameters in pickle file """

    parameters = copy(parameters)
    mode = parameters['BO_Mode']
    del_parameters = ['Kernels', 'Target_Mode', 'Optimization_Mode',
                      'Variance', 'Input_Data', 'Mean_Functions',
                      'LH_Variance', 'Share_Target',
                      'Sampler', 'Sampler_Prior_Var', 'Sampler_Prior_Mean',
                      'Error_Mode', 'BO_Mode', 'Discrete_Mode',
                      'Parameters_Trained', 'Fidelity_Exploration']
    parameters = delete_dict_entries(parameters, del_parameters)

    del_parameters = ['Runtime', 'Dataset', 'Optimized_Likelihood']
    for i, parameter in enumerate(del_parameters):
        del_parameters[i] = parameter
    parameters = delete_dict_entries(parameters, del_parameters)

    del_parameters = []
    ard = parameters['ARD']
    del_parameters.append('ARD')
    lh_variance =\
        parameters['Optimized_LH_Variance'][0]
    del_parameters.append('Optimized_LH_Variance')
    mean_function =\
        parameters['Optimized_Mean_Function'][0]
    del_parameters.append('Optimized_Mean_Function')

    kernel_transform = parameters['Kernel_Transform']
    if kernel_transform is not False:
        idx_comma = kernel_transform.find('_')
        kernel_transform = [float(kernel_transform[0: idx_comma]),
                            float(kernel_transform[idx_comma + 1:])]
    del_parameters.append('Kernel_Transform')

    parameters = delete_dict_entries(parameters, del_parameters)

    if mode == 'MFB-MES' or mode == 'GP-BUCB-Dimension':
        del parameters['Optimized_Model']
        kernel, lengthscale, variance = \
            get_mes_fidelity_kernel_parameters(parameters)
    elif mode == 'GP-BUCB':
        variance = parameters['Optimized_Variance'][0]
        lengthscale = parameters['Optimized_Lengthscales']
        idx_kernel =\
            parameters['Optimized_Model'][0].find('_')
        kernel =\
            parameters['Optimized_Model'][0][0:idx_kernel]
    else:
        raise ValueError(mode + ' is not a valid BO mode')

    kernel_parameters = {'Kernel_Transform': kernel_transform,
                         'Mean_Function': mean_function,
                         'LH_Variance': lh_variance,
                         'ARD': ard,
                         'Variance': variance,
                         'Lengthscale': lengthscale,
                         'Kernel': kernel}

    pickle_file = output_folder + filename
    with open(pickle_file + '.pickle', 'wb') as handle:
        pickle.dump(kernel_parameters, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def get_mes_fidelity_kernel_parameters(parameters):
    """gets the kernel parameters for mes fideility algorithm"""

    lookup_kernels = {'rbf': 'RBF', 'matern12': 'Matern12',
                      'matern32': 'Matern32', 'matern52': 'Matern52'}

    kernel = []
    lengthscale = []
    variance = []
    kernel_count = 1
    while(True):
        for key in parameters:
            idx_number = key.find('_') + 1
            if int(key[idx_number]) == kernel_count:
                lengthscale.append(
                    parameters[key]['Optimized_Lengthscales'])
                if isinstance(parameters[key]['Optimized_Variance'], list):
                    variance.append(parameters[key]['Optimized_Variance'][0])
                else:
                    variance.append(parameters[key]['Optimized_Variance'])
                kernel.append(lookup_kernels[key[0:idx_number - 1]])
                del_key = key
        del parameters[del_key]
        if not parameters:
            break
        else:
            kernel_count = kernel_count + 1
    return kernel, lengthscale, variance
