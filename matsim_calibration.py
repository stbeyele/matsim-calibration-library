"""Reads the experiment and config files and executes exmperiments """
import os
import sys
import time
import shutil
import yaml  # install PyYAML
from experiment.runner import Run_Experiments


if __name__ == '__main__':

    with open('config.txt', 'r') as f:
        config = yaml.load(f)
    with open('simulator.txt', 'r') as f:
        simulator = yaml.load(f)
    config.update(simulator)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    config['output_dir'] = config['output_dir'] + timestr + '/'
    config['temp_dir'] = config['temp_dir'] + timestr + '/'

    with open('experiments.txt', 'r') as f:
        experiments = yaml.load(f)

    experiment_runner = Run_Experiments(config)

    files = []
    logger = {}

    experiment_number = len(experiments.keys())

    for experiment_id in range(experiment_number):
        experiment = 'experiment' + str(experiment_id + 1)
        output_dir = config['output_dir'] + experiment + '/'
        experiment = experiments[experiment]

        if experiment['Mode'] == 'Data_Generation':
            del experiment['Mode']
            filename = experiment_runner.fidelities(experiment, output_dir)

        elif experiment['Mode'] == 'Analyze_Output':
            del experiment['Mode']
            filename = experiment_runner.analyze_output(experiment, output_dir)

        elif experiment['Mode'] == 'BO_Batch':
            del experiment['Mode']
            filename = experiment_runner.bayesian_batch(experiment, output_dir)

        elif experiment['Mode'] == 'BO_Benchmark':
            del experiment['Mode']
            filename = experiment_runner.bo_benchmark(experiment, output_dir)

        elif experiment['Mode'] == 'Test_Error_Functions':
            del experiment['Mode']
            filename = experiment_runner.test_error_functions(experiment,
                                                              output_dir)

        elif experiment['Mode'] == 'Model_Optimization':
            del experiment['Mode']
            filename = experiment_runner.optimize_bayesian(
                experiment, output_dir)

        elif experiment['Mode'] == 'BO_Dim_Reduction':
            del experiment['Mode']
            filename = experiment_runner.bo_reduced_dim(experiment, output_dir)

        else:
            sys.exit('Mode not found')

        files.append(filename)
        logger[filename] = experiment

        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])

    if os.path.isfile(config['log_dir'] + '/log.txt') is True:
        with open(config['log_dir'] + '/log.txt', 'r') as f:
            log_file = yaml.load(f)
    else:
        log_file = {}

    del config['temp_dir']

    if config['delete_output_history'] is True:

        if os.path.exists(config['output_dir']):
            shutil.rmtree(config['output_dir'])

    log_file.update(logger)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    config['Filenames with config'] = files
    log_file['---Configuration---' + timestr] = config
    with open(config['log_dir'] + '/log.txt', 'w') as f:
        yaml.dump(log_file, f, default_flow_style=False)

# get r2 error to ground truth
# run same setting for different iteration number!!!
