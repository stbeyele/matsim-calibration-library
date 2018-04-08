"""This module analyzes the MATSim output files"""
import csv
import sys
import os
import gzip
from copy import copy
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np
import pathos.pools as pp


class Result_Analyser(object):

    def __init__(self, parameters):

        target_mode = parameters['Target_Mode']
        self.analyzer = None

        if target_mode == 'Mode_Share':
            self.analyzer = Result_Analyser_Mode_Share(parameters)
        elif target_mode == 'Travel_Times':
            self.analyzer = Result_Analyser_Travel_Times()
        else:
            raise ValueError(target_mode + ' is not a valid error mode')

    def get_errors(self, batch_size, paths):

        errors = self.analyzer.get_target_distribution(batch_size, paths)

        return errors

    def get_runtimes(self, paths):

        runtimes = get_runtime(paths)

        return runtimes

    def get_bins(self):

        return self.analyzer.get_bins()


class Result_Analyser_Mode_Share(object):
    """ Gets mode counts from the events file"""

    def __init__(self, parameters):

        self.modes = parameters['Modes']
        self.paths = []

    def mode_share(self, path):
        """ Get mode share distribution from events file"""

        shares = {}
        for share in self.modes:
            shares[share] = 0

        # Open events
        tree = open_xml(path)
        root = tree.getroot()

        # Count
        for ev in root.findall("./event[@type='departure']"):
            for mode in self.modes:
                if mode.lower() == 'walk':
                    event_mode = ['walk', 'transit_walk']
                else:
                    event_mode = [mode.lower()]
                if ev.attrib['legMode'] in event_mode:
                    shares[mode] += 1
        return shares

    def init_target_distribution(self):
        """Init Dictionary for saving the mode share distribution"""

        mode_shares = {}
        for mode in self.modes:
            mode_shares[mode] = []

        return mode_shares

    def get_target_distribution(self, batch_size, paths):
        """ Returns mode share distribution """

        self.paths = paths
        mode_shares = self.init_target_distribution()

        batch_share = self.get_target_distribution_batch(batch_size)
        for key, value in batch_share.iteritems():
            mode_shares[key] = mode_shares[key] + value

        return mode_shares

    def calc_mode_share(self, i):
        """Calculates mode share from mode counts"""

        mode_share = self.mode_share(
            self.paths[i]['output_dir'] + self.paths[i]['events_file'])

        total = np.sum(mode_share.values())
        for key, value in mode_share.iteritems():
            mode_share[key] = float(value) / float(total)
        mode_share['Index'] = i

        return mode_share

    def get_target_distribution_batch(self, batch_size):
        """Batch version which can analyse several files in parallel"""

        batch_shares = {}
        for mode in self.modes:
            batch_shares[mode] = [0 for idx in range(len(self.paths))]

        index = range(len(self.paths))
        pool = pp.ProcessPool(batch_size)
        mode_shares = pool.map(self.calc_mode_share, index)
        pool.close()
        pool.join()
        pool.clear()

        for mode_share in mode_shares:
            index = mode_share['Index']
            del mode_share['Index']
            for key, value in mode_share.iteritems():
                batch_shares[key][index] = value

        return batch_shares

    def get_target_distribution_bo(self, paths):
        """Returns target distribution for Bayesian Optimization run"""

        self.paths = paths

        batch_size = len(self.paths)
        if batch_size > 1:
            final_shares = self.get_target_distribution_batch(batch_size)
        else:
            final_shares = self.calc_mode_share(0)
            del final_shares['Index']

        modes = copy(self.modes)
        shares = dict_to_array(final_shares, modes)

        return shares

    def get_bins(self):
        """Returns all avalable modes"""

        return self.modes


def get_runtime(paths):
    """ Returns the Runtimes of the Runs which are read fro MATSim output"""

    timeStrings = {}
    for path in paths:

        file_path = path['output_dir'] + path['stopwatch_file']
        with open(file_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            stopwatch = list(reader)

        start_col = stopwatch[0].index('BEGIN iteration')
        end_col = stopwatch[0].index('END iteration')
        start_time = datetime.strptime(stopwatch[1][start_col], '%H:%M:%S')
        end_time = datetime.strptime(stopwatch[-1][end_col], '%H:%M:%S')
        run_time = (end_time - start_time).seconds
        run_time = timedelta(seconds=run_time)
        timeStrings.update({path['run_counter']: str(run_time)})

    return timeStrings


class Result_Analyser_Travel_Times(object):
    """Gets travel time distribution from the tripdurations file"""

    def __init__(self):

        self.travel_times = ['0+', '5+', '10+', '15+', '20+', '25+', '30+',
                             '35+', '40+', '45+', '50+', '55+', '60+']
        self.paths = None
        self.offset = 0

    def init_target_distribution(self):
        """ Initialize Dictionary to store the travel time distribution"""

        travel_times = {}
        for travel_time in self.travel_times:
            travel_times[travel_time] = []

        return travel_times

    def get_target_distribution(self, batch_size, paths):
        """Gets the travel time distribution from the tripduration file"""

        travel_times = self.init_target_distribution()

        for path in paths:

            trip_file = path['output_dir'] + path['tripduration_file']

            csv_file = []
            with open(trip_file, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for row in reader:
                    csv_file.append(row)
            file_size = len(csv_file)

            travel_time = self.init_target_distribution()
            for j in range(1, file_size - 2):
                row_length = len(csv_file[j])
                for k in range(1, row_length):
                    travel_time[csv_file[0][k]].append(int(csv_file[j][k]))

            sum_travel_times = 0
            for key in travel_times:
                sum_travel_times = sum_travel_times + np.sum(travel_time[key])
                travel_times[key].append(np.sum(travel_time[key]).tolist())

            for key in travel_times:
                travel_times[key][-1] = (float(travel_times[key][-1]) /
                                         float(sum_travel_times))
        return travel_times

    def get_bins(self):
        """Returns the travel time modes"""

        return self.travel_times

    def get_target_distribution_bo(self, paths):
        """Returns travel time distribution for Bayesian Optimization"""

        travel_times = self.init_target_distribution()
        travel_times = self.get_target_distribution(0, paths)
        share_modes = copy(self.travel_times)
        shares = dict_to_array(travel_times, share_modes)

        return shares


def open_xml(path):
    """ Open xml and xml.gz files into ElementTree """

    if path.endswith('.gz'):
        xml_file = ET.parse(gzip.open(path))
    else:
        xml_file = ET.parse(path)

    return xml_file

def dict_to_array(dict_shares, modes):
    """Convert dictionary to ultidimensional array"""

    shares = np.asarray(dict_shares[modes.pop(0)])
    for mode in modes:
        shares = np.vstack((shares, np.asarray(dict_shares[mode])))

    return shares
