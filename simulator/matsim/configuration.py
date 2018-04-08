"""Generates all necessary configuration files for running MATSim"""
import os
import sys
import yaml
import numpy as np
from numpy.random import RandomState
from utilities.io_utilities import check_create_dir


with open('config.txt', 'r') as f:
    config = yaml.load(f)
with open('simulator.txt', 'r') as f:
    simulator = yaml.load(f)
import jnius_config
if 'log4j_config' in config:
    log4j_config_file = '-Dlog4j.configuration=file:' + config['log4j_config']
    jnius_config.add_options(
        '-Xss2m', '-Djava.awt.headless=true', log4j_config_file)
else:
    jnius_config.add_options('-Xss2m', '-Djava.awt.headless=true')
jnius_config.set_classpath('.', simulator['matsim_dir'])
from jnius import autoclass


class MatSimConfigurator(object):
    """Generated the MATSim configuration file"""

    def __init__(self, dir_scenario, dir_temp, dir_output):

        self.ConfigUtils = autoclass('org.matsim.core.config.ConfigUtils')
        self.ConfigWriter = autoclass('org.matsim.core.config.ConfigWriter')
        self.ScenarioUtils = autoclass(
            'org.matsim.core.scenario.ScenarioUtils')
        self.OverwriteFileSetting = autoclass(
            'org.matsim.core.controler.'
            'OutputDirectoryHierarchy$OverwriteFileSetting')
        self.TransportMode = autoclass('org.matsim.api.core.v01.TransportMode')
        self.PopulationWriter = autoclass(
            'org.matsim.core.population.io.PopulationWriter')
        self.dir_scenario = dir_scenario
        self.dir_temp = dir_temp
        self.dir_output = dir_output
        self.config = self.ConfigUtils.createConfig()

    def get_config(self):
        """Returns the MATSim configuration object"""

        return self.config

    def write_config_file(self, config_name):
        """Writes the configuration to a file"""

        path = os.path.join(self.dir_temp, 'configs', config_name)
        check_create_dir(os.path.join(self.dir_temp, 'configs'))
        output = self.ConfigWriter(self.config)
        output.write(path)

        return path

    def load_config_file(self, config_name):
        """Loads a configuration file"""

        path = os.path.join(self.dir_scenario, config_name)
        self.ConfigUtils.loadConfig(self.config, path)

    def set_network_file(self, netwrok_name):
        """Sets a newtork file path"""

        path = os.path.join(self.dir_scenario, netwrok_name)
        self.config.network().setInputFile(path)
        print self.config.network().getInputFile()

    def set_plans_file(self, plans_name):
        """Sets a plans file path"""

        path = os.path.join(self.dir_scenario, plans_name)
        self.config.plans().setInputFile(path)
        print self.config.plans().getInputFile()

    def set_attributes_file(self, attributes_name):
        """Sets an attributes file path"""

        path = os.path.join(self.dir_scenario, attributes_name)
        self.config.plans().setInputPersonAttributeFile(path)
        print self.config.plans().getInputPersonAttributeFile()

    def set_facilites_file(self, facilites_name):
        """Sets a facilities file path"""

        path = os.path.join(self.dir_scenario, facilites_name)
        self.config.facilities().setInputFile(path)
        print self.config.facilities().getInputFile()

    def set_transitschedule_file(self, transitschedule_name):
        """Sets a transitschedule file path"""
        path = os.path.join(self.dir_scenario, transitschedule_name)
        self.config.transit().setTransitScheduleFile(path)
        print self.config.transit().getTransitScheduleFile()

    def set_vehicle_file(self, vehicle_name):
        """Sets a vehicle file path"""
        path = os.path.join(self.dir_scenario, vehicle_name)
        self.config.transit().setVehiclesFile(path)
        print self.config.transit().getVehiclesFile()

    def set_lastiteration(self, last_iteration):
        """Sets iteration number"""

        self.config.controler().setLastIteration(int(last_iteration))
        print self.config.controler().getLastIteration()

    def set_writeinterval(self, writeinterval):
        """Sets output files writing interval"""

        self.set_writeeventsinterval(writeinterval)
        self.set_writeplansinterval(writeinterval)
        self.set_writesnapshosinterval(writeinterval)
        self.set_writecountsinterval(writeinterval)
        self.set_writelinkstatsinterval(writeinterval)

    def set_writeeventsinterval(self, eventsinterval):
        """Sets events files writing interval"""

        self.config.controler().setWriteEventsInterval(eventsinterval)
        print self.config.controler().getWriteEventsInterval()

    def set_writesnapshosinterval(self, snapshotsinterval):
        """Sets snapshots files writing interval"""
        self.config.controler().setWriteSnapshotsInterval(snapshotsinterval)
        print self.config.controler().getWriteSnapshotsInterval()

    def set_writecountsinterval(self, countsinterval):
        """Sets counts files writing interval"""

        self.config.counts().setWriteCountsInterval(countsinterval)
        print self.config.counts().getWriteCountsInterval()

    def set_writelinkstatsinterval(self, linkstatsinterval):
        """Sets linkstats files writing interval"""

        self.config.linkStats().setWriteLinkStatsInterval(linkstatsinterval)
        print self.config.linkStats().getWriteLinkStatsInterval()

    def set_writeplansinterval(self, plansinterval):
        """Sets plans files writing interval"""
        self.config.controler().setWritePlansInterval(plansinterval)
        print self.config.controler().getWritePlansInterval()

    def set_popfactor(self, factor, population_name, seed, mode):
        """Downsamples population to a given fraction"""

        self.set_flowcapfactor(factor)
        self.set_storagecapfactor(factor)

        if factor < 1.0:
            path = os.path.join(self.dir_temp, 'populations', population_name)
            check_create_dir(os.path.join(self.dir_temp, 'populations'))
            scenario = self.ScenarioUtils.loadScenario(self.config)
            persons = scenario.getPopulation().getPersons()
            population_generator = PopulationSampler(self.config)

            print mode

            if mode == 'Random':
                print 'Random Mode'
                population_generator.get_new_population_rand(
                    persons, seed, factor)
            else:
                print 'Representative Mode'
                population_generator.get_new_population_stat(
                    persons, seed, factor)

            population = population_generator.get_population()
            self.PopulationWriter(population).write(path)
            self.config.plans().setInputFile(path)

        else:
            path = self.config.plans().getInputFile()

        scenario = self.ScenarioUtils.loadScenario(self.config)
        population = scenario.getPopulation()

        return path

    def set_flowcapfactor(self, flowcapfactor):
        """Set flow capacitiy factor"""

        self.config.qsim().setFlowCapFactor(flowcapfactor)
        print self.config.qsim().getFlowCapFactor()

    def set_storagecapfactor(self, storagecapfactor):
        """Set storage capacity factor"""

        self.config.qsim().setStorageCapFactor(storagecapfactor)
        print self.config.qsim().getStorageCapFactor()

    def set_car_modeparams(self, constant):
        """Set car constant"""

        CarSet = self.config.planCalcScore().getOrCreateModeParams(
            self.TransportMode.car)
        CarSet.setConstant(constant)
        print CarSet.getConstant()

    def set_walk_modeparams(self, constant):
        """Set walk constant"""

        WalkSet = self.config.planCalcScore().getOrCreateModeParams(
            self.TransportMode.walk)
        WalkSet.setConstant(constant)
        print WalkSet.getConstant()

    def set_pt_modeparams(self, constant):
        """Set pt constant"""

        ptSet = self.config.planCalcScore().getOrCreateModeParams(
            self.TransportMode.pt)
        ptSet.setConstant(constant)
        print ptSet.getConstant()

    def set_mode_constant(self, constant, mode):
        """Set pt constant"""
        pos = mode.find('_')
        mode = mode[0:pos]
        ptSet = self.config.planCalcScore().getOrCreateModeParams(
            getattr(self.TransportMode, mode.lower()))
        ptSet.setConstant(constant)
        print ptSet.getConstant()

    def set_output_dir(self, output_folder):
        """Set output directory"""

        path = os.path.join(self.dir_output, output_folder)
        self.config.controler().setOutputDirectory(path)
        print self.config.controler().getOutputDirectory()

        return path

    def set_overwritefile(self):
        """Set if outputfiles are overwritten"""

        self.config.controler().setOverwriteFileSetting(
            self.OverwriteFileSetting.deleteDirectoryIfExists)

    def set_nthreads(self, n_cores):
        """Set number of threads used for simulation"""

        self.config.setParam('global', 'numberOfThreads', str(n_cores))
        print self.config.getParam('global', 'numberOfThreads')

        if n_cores > 1:

            n_event = 1
            n_qsim = n_cores - 1

            self.config.qsim().setNumberOfThreads(n_qsim)
            print self.config.qsim().getNumberOfThreads()
            self.config.setParam('parallelEventHandling', 'numberOfThreads',
                                 str(n_event))
            print self.config.parallelEventHandling().getNumberOfThreads()

    def set_runid(self, id):
        """Set runid"""

        self.config.controler().setRunId(id)
        print self.config.controler().getRunId()

    def set_seed(self, seed):
        """Set MATSim seed"""

        self.config.setParam('global', 'randomSeed', str(int(seed)))
        print self.config.getParam('global', 'randomSeed')


class PopulationSampler(object):
    """Downsamples population"""

    def __init__(self, config_file):

        self.config = config_file
        self.PopulationUtils = autoclass(
            'org.matsim.core.population.PopulationUtils')
        self.PersonUtils = autoclass(
            'org.matsim.core.population.PersonUtils')
        self.ArrayList = autoclass('java.util.ArrayList')
        self.population = None

    def get_new_population_stat(self, persons, seed, factor):
        """Generate new population with a given distribution"""

        ran_gen = RandomState(int(seed))
        size = persons.size()
        population = self.PopulationUtils.createPopulation(self.config)
        person_ids = self.ArrayList(persons.keySet())

        person_ids_list = []
        for i in range(size):
            person_ids_list.append(person_ids.get(i))

        person_ids = person_ids_list
        global_distr, max_age = self.get_distribution(persons, person_ids)

        for key, value in global_distr.iteritems():
            global_distr[key] = np.around(value * factor)

        for age in range(max_age):
            person_pool_age = self.get_person_pool(
                persons, person_ids, 'getAge', age)

            for employed in [0, 1]:
                person_pool_employed = self.get_person_pool(
                    persons, person_pool_age, 'isEmployed', employed)

                for sex in ['m', 'f']:
                    person_pool_sex = self.get_person_pool(
                        persons, person_pool_employed, 'getSex', sex)

                    for car_avail in ['always', 'never']:
                        person_pool = self.get_person_pool(
                            persons, person_pool_sex, 'getCarAvail', car_avail)
                        key = (str(age) + '_' + car_avail + '_' +
                               sex + '_' + str(employed))

                        if key in global_distr:

                            if global_distr[key] > 0.0:
                                sampled_ids = ran_gen.choice(
                                    range(0, len(person_pool)),
                                    size=int(global_distr[key]), replace=False)

                                for sampled_id in sampled_ids:
                                    person_id = person_pool[sampled_id]
                                    population.addPerson(
                                        persons.get(person_id))

        self.population = population
        # persons = population.getPersons()
        # person_ids = self.ArrayList(persons.keySet())
        # person_ids_list = []
        # size = persons.size()
        # for i in range(size):
        #     person_ids_list.append(person_ids.get(i))
        # person_ids = person_ids_list
        # global_distr, max_age = self.get_distribution(persons, person_ids)
        # print global_distr

    def get_distribution(self, persons, person_ids):
        """Get population distribution"""

        max_age = 0
        global_distr = {}

        for person_id in person_ids:
            person = persons.get(person_id)
            age = self.PersonUtils.getAge(person)

            if age > max_age:
                max_age = age

            car_avail = self.PersonUtils.getCarAvail(person)
            sex = self.PersonUtils.getSex(person)
            employed = self.PersonUtils.isEmployed(person)
            key = str(age) + '_' + car_avail + '_' + sex + '_' + str(employed)

            if key in global_distr:
                global_distr[key] = global_distr[key] + 1
            else:
                global_distr.update({key: 1})

        return global_distr, max_age

    def get_person_pool(self, persons, person_ids, attribute, value):
        """Generate person pools of persons with similar behavior"""
        person_pool = []
        for person_id in person_ids:
            person = persons.get(person_id)
            if getattr(self.PersonUtils, attribute)(person) == value:
                person_pool.append(person_id)
        # person_pool = []

        # for person_id in person_ids:
        #     person = persons.get(person_id)
        #     plans = person.getPlans()

        #     for i in range(plans.size()):
        #         plan = plans.get(i)
        #         activities = plan.getPlanElements()

        #         for j in range(activities.size()):
        #             activity = activities.get(j)

        #             if isinstance(
        #                 activity, autoclass(
        #                     'org.matsim.core.population.ActivityImpl')):
        #                 start_time = activity.getStartTime()

        #                 if start_time > 0.0:
        #                     m, s = divmod(start_time, 60)
        #                     h, m = divmod(m, 60)
        #                 end_time = activity.getEndTime()

        #                 if end_time > 0.0:
        #                     m, s = divmod(end_time, 60)
        #                     h, m = divmod(m, 60)

        #             else:
        #                 print activity.getMode()

        return person_pool

    def get_new_population_rand(self, persons, seed, factor):
        """Generate new population with random sampling"""

        size = persons.size()
        population = self.PopulationUtils.createPopulation(self.config)
        person_ids = self.ArrayList(persons.keySet())

        ran_gen = RandomState(int(seed))
        sampled_ids = ran_gen.choice(
            range(0, size), size=int(size * factor), replace=False)

        for sampled_id in sampled_ids:
            person_id = person_ids.get(sampled_id)
            population.addPerson(persons.get(person_id))

        self.population = population

    def get_population(self):
        """Retrun population"""

        return self.population
