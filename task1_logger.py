import numpy as np
import os

SIGNIF = 3


class Logger:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.log_file = "Logs/Task1/{name:s}/log.txt".format(name=self.experiment_name)

        directory = "Logs/Task1/{name:s}".format(name=experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file = open(self.log_file, "a")
        file.write("----Start new run----\n")


    def log_results(self, fitness):
        file = open(self.log_file, "a")
        file.write("%.{n}f,%.{n}f\n".format(n=SIGNIF) %(self.__avarage__(fitness), self.__max__(fitness)))

    def __avarage__(self, np_array):
        return np_array.mean()

    def __max__(self, np_array):
        return np_array.max()
