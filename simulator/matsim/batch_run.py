"""Run a batch of simulations"""
import sys
import os
from subprocess import Popen
from copy import copy
from simulator.matsim.configuration import MatSimConfigurator
from utilities.io_utilities import check_if_files_exist, get_value


class Batch_Run(object):
    """ Run a batch of simulations"""

    def __init__(self, simulator_paths, parameters):

        self.paths = simulator_paths
        if 'Batch_Size' in parameters:
            self.batch_size = parameters['Batch_Size']
        else:
            self.batch_size = 1
        if 'RAM_Amount' in parameters:
            self.ram_amount = parameters['RAM_Amount']
        else:
            self.ram_amount = 4096

        self.batch_paths = []
        self.matsim_dir = ''
        self.log4j_config_file = ''
        self.global_run_counter = 0
        self.main_class = None

    def update_batch_size(self, batch_size):
        """update batch size"""

        self.batch_size = batch_size

    def prepare_batches(self, parameters, offset, run_offset):
        """Perpare all neccessary parameters and inputfiles for batch run"""

        scenario_dir = self.paths['scenario_dir']
        self.main_class = ' ' + self.paths['main_class'] + ' '
        self.matsim_dir = self.paths['matsim_dir']
        temp_dir = self.paths['temp_dir']
        output_dir = self.paths['output_dir']

        if not self.paths['log4j_config'] is False:
            self.log4j_config_file = (
                ' -Dlog4j.configuration=file:' + self.paths['log4j_config'])

        config_name = self.paths['config_name']
        network_name = self.paths['network_name']
        population_name = self.paths['population_name']
        facilites_name = self.paths['facilities_name']

        self.batch_paths = []
        for i in range(0, self.batch_size):

            print parameters

            pop_fraction = parameters['Population_Fraction'][i + offset]
            pop_mode = parameters['Population_Mode'][i + offset]
            iterations = parameters['Iterations'][i + offset]
            seed_pop = parameters['Seed'][i + offset]
            matsim_seed = parameters['Seed'][i + offset]
            output_folder = 'Scenario_' + str(i + offset + run_offset) + '/'
            subpopulation_name = ('Population_' +
                                  str(i + offset + run_offset) +
                                  '_' + str(pop_fraction) + '.xml.gz')
            config_mod_name = 'Config_' + str(i + offset + run_offset) + \
                '.xml'
            run_id = str(i + offset)

            Sim = MatSimConfigurator(scenario_dir, temp_dir, output_dir)
            Sim.load_config_file(config_name)
            Sim.set_overwritefile()
            Sim.set_network_file(network_name)
            Sim.set_plans_file(population_name)
            Sim.set_facilites_file(facilites_name)
            if 'transitschedule_name' in self.paths:
                Sim.set_transitschedule_file(
                    self.paths['transitschedule_name'])
            if 'vehicles_name' in self.paths:
                Sim.set_vehicle_file(self.paths['vehicles_name'])
            Sim.set_lastiteration(iterations)
            Sim.set_writeinterval(iterations)
            Sim.set_runid(run_id)
            Sim.set_seed(matsim_seed)
            if 'population_attributes_name' in self.paths:
                Sim.set_attributes_file(
                    self.paths['population_attributes_name'])

            for parameter in parameters['Parameters']:
                Sim.set_mode_constant(parameters[parameter][i + offset],
                                      parameter)

            population_path = Sim.set_popfactor(
                pop_fraction, subpopulation_name, seed_pop, pop_mode)
            output_folder = Sim.set_output_dir(output_folder)
            config_path = Sim.write_config_file(config_mod_name)
            events_file = run_id + '.output_events.xml.gz'
            stopwatch_file = run_id + '.stopwatch.txt'
            tripduration_file = 'ITERS/it.' + \
                str(parameters['Iterations'][offset + i]) + '/' + run_id +\
                '.' + str(parameters['Iterations'][offset + i]) + \
                '.tripdurations.txt'
            self.batch_paths.append({
                'output_dir': output_folder,
                'config_path': config_path,
                'population_path': population_path,
                'events_file': events_file,
                'stopwatch_file': stopwatch_file,
                'tripduration_file': tripduration_file,
                'run_counter': copy(self.global_run_counter)
            })
            self.global_run_counter = self.global_run_counter + 1

    def run_batches(self):
        """Run batch of simulations"""

        commands = []
        for idx in range(self.batch_size):
            command = self.generate_command(idx)
            commands.append(command)

        processes = [Popen(cmd, shell=True) for cmd in commands]
        for p in processes:
            p.wait()

        return self.batch_paths

    def generate_command(self, idx):
        """Generating Java command"""
        command = ('java -Xss2m -Djava.awt.headless=true -Xmx' +
                   str(int(self.ram_amount)) + 'm -cp ' +
                   self.matsim_dir + self.log4j_config_file +
                   self.main_class +
                   self.batch_paths[idx]['config_path'])
        return command

    def run_check_batches(self, parameters, offset, run):
        """Run batches and check if output files are generated"""

        start_analysis = False
        while start_analysis is False:
            self.prepare_batches(parameters, offset,
                                 run * self.batch_size)
            return_values = self.run_batches()
            output_dirs = []
            for return_value in return_values:
                output_dirs.append(return_value['output_dir'] +
                                   return_value['events_file'])
            start_analysis = check_if_files_exist(output_dirs)

        return return_values
