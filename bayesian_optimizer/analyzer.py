"""Module to plot Bayesian Optimization performance"""
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn




class BO_Analyser(object):

    def __init__(self, batch_size, filename, folder, global_min):

        self.batch_size = batch_size
        self.filename = filename
        self.folder = folder
        self.global_min = global_min
        self.specific_name = None
        plt.clf()

    def plot_bo(self, y, y_opt):
        """Generate plot runs vs accuracy and save it in a file"""

        y = y * -1
        runs = y.shape[0] / self.batch_size
        size = np.linspace(0, y.shape[0], runs + 1)

        min_values = []
        global_min = float('inf')
        for i in range(size.shape[0] - 1):
            min_value = np.min(y[int(size[i]):int(size[i + 1])])
            if min_value < global_min and self.global_min:
                global_min = min_value
            else:
                global_min = min_value
            min_values.append(global_min)
        run = range(len(min_values))

        plt.plot(run, min_values, '-x')
        plt.axhline(y_opt, color='r')
        plt.ylabel('Error Measure')
        plt.xlabel('Run Number')
        plt.xticks(map(int, run))

        self.specific_name = '_batch'

    def save_figure(self):
        plt.savefig(self.folder + self.filename +
                    self.specific_name + '.png')
        plt.clf()

    #  integrate plot library as well
