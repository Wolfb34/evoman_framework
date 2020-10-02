import numpy as np
import os
from task1_constants import *

SIGNIF = 3


class Logger:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.log_file_avg = "Logs/Task1/{name:s}/log_avg.txt".format(name=self.experiment_name)
        self.log_file_max = "Logs/Task1/{name:s}/log_max.txt".format(name=self.experiment_name)
        self.log_file_ind = "Logs/Task1/{name:s}/log_ind.txt".format(name=self.experiment_name)
        self.log_file_div = "Logs/Task1/{name:s}/log_div.txt".format(name=self.experiment_name)


        directory = "Logs/Task1/{name:s}".format(name=experiment_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_avg = open(self.log_file_avg, "a")
        file_max = open(self.log_file_max, "a")
        file_div = open(self.log_file_div, "a")

        file_avg.write("\n")
        file_max.write("\n")
        file_div.write("\n")


    def log_results(self, fitness, population):
        file_avg = open(self.log_file_avg, "a")
        file_max = open(self.log_file_max, "a")
        file_div = open(self.log_file_div, "a")


        file_avg.write("%.{n}f, ".format(n=SIGNIF) % (self.__average__(fitness)))
        file_max.write("%.{n}f, ".format(n=SIGNIF) % (self.__max__(fitness)))
        file_div.write("%.{n}f, ".format(n=SIGNIF) % (self.__Estimated_Hamming_Diversity__(population)))


    def log_individual(self, individual_gain):
        file_ind = open(self.log_file_ind, "a")
        file_ind.write("%.{n}f, ".format(n=SIGNIF) % individual_gain)

    #returns the hamming diversity as a measure from 0 to 100. With 100 being maximum diversity.
    def __Estimated_Hamming_Diversity__(self, population):
        mean_individual = np.mean(population, axis=0)

        n_vars = mean_individual.shape[0]
        total_difference = np.zeros(n_vars)

        #calculate total difference between individuals and mean individual
        for individual in population:
            for i in range(n_vars):
                total_difference[i] += abs(mean_individual[i] - individual[i])

        #calculate mean/average distance between individual and mean individudal
        mean_difference = np.zeros(n_vars)
        for i in range(n_vars):
            mean_difference[i] = total_difference[i] / NPOP

        estimated_hamming_diversity = round( 100 * np.mean(mean_difference), 3)

        return estimated_hamming_diversity

    def __average__(self, np_array):
        return np_array.mean()

    def __max__(self, np_array):
        return np_array.max()
