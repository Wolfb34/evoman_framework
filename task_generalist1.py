################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys, os
import numpy as np
sys.path.insert(0, 'evoman')
from task1_selection import Selection
from task1_mutation import Mutation
from task1_logger import Logger
from task1_recombination import Recombination
from task1_initialization import Initialization
from task1_evaluation import Evaluation
from demo_controller import player_controller
from environment import Environment
from task1_constants import *


class Generalist1:

    def __init__(self,  enemies):
        self.enemies = enemies
        self.experiment_name = 'task2_generalist1_enemies_{}'.format(enemies)

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        self.env = Environment(experiment_name=self.experiment_name, level=2,
                          player_controller=player_controller(N_HIDDEN_NEURONS),
                          multiplemode="yes",
                          enemies=enemies,
                          speed="fastest")

        self.n_vars = (self.env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
        self.rot_size = int((self.n_vars * (self.n_vars - 1)) / 2)
        self.dev = np.random.uniform(0, INIT_SD, (NPOP, self.n_vars))
        self.rot = np.random.uniform(-np.pi, np.pi, (NPOP, self.rot_size))

        self.init = Initialization(DOM_L, DOM_U)
        self.evaluator = Evaluation(self.env, enemies, SHARE_SIZE)
        self.selector = Selection()
        self.logger = Logger(self.experiment_name)
        self.recombinator = Recombination()
        self.mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)

    def __run_best_against_all__(self):
        player_array, enemy_array = [], []
        for i in range(1, 9):
            self.env.update_parameter('enemies', [i])
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=np.array(self.best_individual[0]))
            player_array.append(player_life)
            enemy_array.append(enemy_life)
        return sum(player_array) - sum(enemy_array)

    def store_best_champion(self, pop, fit, gen):
        if fit.max() > self.highest_fitness:
            self.best_individual = self.selector.select_best_n(pop,fit,1)
            self.best_gen = gen
            self.highest_fitness = fit.max()

    def run(self):
        population = self.init.uniform_initialization(NPOP, self.n_vars)

        self.best_gen = 0
        self.highest_fitness = -100000
        self.best_individual = None

        for generation in range(1,NGEN+1):
            print("EVALUATION GENERATION %d OF %d \n" %(generation, NGEN))

            fitness_list = self.evaluator.simple_generalist_eval(population)

            '''Log fitness'''
            self.logger.log_results(fitness_list, population)
            self.store_best_champion(population, fitness_list,generation)

            '''create next gen'''
            if generation != NGEN:
                parents = self.selector.tournament_percentage(population, fitness_list)
                survivors = self.selector.select_best_percentage(population, fitness_list, BEST_SURVIVOR_PERCENTAGE)

                '''create children'''
                children = self.recombinator.blend(parents, NPOP-len(survivors))

                #children, self.dev, self.rot = self.mutator.correlated_mutation(children, self.dev, self.rot)
                children, self.dev = self.mutator.uncorrelated_mutation_n_step_size(children, self.dev)

                '''combine survivors and children'''
                population = np.append(children, survivors, axis=0)

        # Run the best individual of all generations
        print("The best fitness was in generation %d and had a fitness of %.3f" %(self.best_gen, self.highest_fitness))

        individual_gain = []
        for run in range(5):
            individual_gain.append(self.__run_best_against_all__())

        average_ig = sum(individual_gain)/len(individual_gain)
        self.logger.log_individual(average_ig)




group1 = Generalist1(GROUP1)
group1.run()
group2 = Generalist1(GROUP2)
group2.run()
