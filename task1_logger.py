import numpy as np
import os

SIGNIF = 3


class Logger:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.log_file_avg = "Logs/Task1/{name:s}/log_avg.txt".format(name=self.experiment_name)
        self.log_file_max = "Logs/Task1/{name:s}/log_max.txt".format(name=self.experiment_name)


        directory = "Logs/Task1/{name:s}".format(name=experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_avg = open(self.log_file_avg, "a")
        file_max = open(self.log_file_max, "a")

        file_avg.write("\n")
        file_max.write("\n")

    def log_results(self, fitness):
        file_avg = open(self.log_file_avg, "a")
        file_max = open(self.log_file_max, "a")

        file_avg.write("%.{n}f, ".format(n=SIGNIF) % (self.__average__(fitness)))
        file_max.write("%.{n}f, ".format(n=SIGNIF) % (self.__max__(fitness)))

    def __average__(self, np_array):
        return np_array.mean()

    def __max__(self, np_array):
        return np_array.max()
