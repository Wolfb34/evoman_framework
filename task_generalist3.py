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
import itertools

class Generalist3:

    def __init__(self,  enemies):
        self.enemies = enemies

        experiment_num = 0
        while True:
            self.experiment_name = 'task2_generalist3_enemies_{}_{}'.format(enemies, experiment_num)
            if not os.path.exists(self.experiment_name):
                break
            experiment_num += 1
        os.makedirs(self.experiment_name)

        self.env = Environment(experiment_name=self.experiment_name, level=2,
                               player_controller=player_controller(N_HIDDEN_NEURONS),
                               enemies=[enemies[0]],
                               speed="fastest")

        self.n_vars = (self.env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
        self.rot_size = int((self.n_vars * (self.n_vars - 1)) / 2)
        self.dev = np.random.uniform(0, INIT_SD, (NPOP, self.n_vars))
        self.rot = np.random.uniform(-np.pi, np.pi, (NPOP, self.rot_size))
        self.saw = np.ones(np.shape(enemies))

        self.init = Initialization(DOM_L, DOM_U)
        self.evaluator = Evaluation(self.env, enemies, SHARE_SIZE)
        self.selector = Selection()
        self.logger = Logger(self.experiment_name)
        self.recombinator = Recombination()
        self.mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)

    def __compare_to_ultimate__(self, individual_gain, wins, champion):
        ultimate_performance_file = open("Logs/Task1/UltimateChampion/UltimatePerformance.txt", "r+")
        ultimate_performance, ultimate_wins = eval(ultimate_performance_file.read())

        #declare new ultimate champion if either has more wins or same amount of wins and greater performance
        if wins >= ultimate_wins and ((wins > ultimate_wins) or (individual_gain > ultimate_performance)):
            ultimate_file = open("Logs/Task1/UltimateChampion/UltimateChampion.txt", "w")
            ultimate_file.write(np.array_str(champion))

            ultimate_performance_file.seek(0)
            ultimate_performance_file.truncate()
            ultimate_performance_file.write("".join(map(str, (individual_gain,", ", wins))))


    def __run_best_against_all__(self):
        player_array, enemy_array = [], []
        wins = 0
        for i in range(1, 9):
            self.env.update_parameter('enemies', [i])
            _, player_life, enemy_life, _ = self.env.play(pcont=np.array(self.best_individual[0]))
            player_array.append(player_life)
            enemy_array.append(enemy_life)
            if enemy_life == 0:
                wins += 1
        return (sum(player_array) - sum(enemy_array)), wins

    def __stepwise_adaption_of_weights__(self, saw):
        fitness_array = np.zeros(8)
        for i in range(1, 9):
            self.env.update_parameter('enemies', [i])
            fitness, _, _, _ = self.env.play(pcont=np.array(self.best_individual[0]))
            fitness_array[i-1] = fitness

        fitness_indices = np.argsort(fitness_array)
        max_index = 7
        min_index = 0
        while max_index != min_index:
            if saw[fitness_indices[max_index]] <= 0.11:
                max_index -= 1
            elif saw[fitness_indices[min_index]] >= 1.89:
                min_index += 1
            else:
                saw[fitness_indices[max_index]] -= 0.1
                saw[fitness_indices[min_index]] += 0.1
                break
        return saw


    def __share_of__(self, ind1, ind2):
        dist = np.linalg.norm(ind1 - ind2)
        if dist > SHARE_SIZE:
            return 0
        return 1 - dist / SHARE_SIZE

    def __share_fitness__(self, pop, fitness):
        new_fitness = []
        length = len(pop)
        for i in range(length):
            divisor = sum([self.__share_of__(pop[i], pop[j]) for j in range(length)])
            new_fitness.append(fitness[i] / divisor)
        return np.array(new_fitness)

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

        for generation in itertools.count(start=1):
            print("\nEVALUATION GENERATION %d \n" % generation)

            fitness_list = self.evaluator.sharing_generalist_eval(population, saw=self.saw)

            '''Log fitness'''
            self.logger.log_results(fitness_list, population)
            self.store_best_champion(population, fitness_list, generation)
            min_fitness = np.amin(fitness_list)
            print("Fitness before normalization:\n" + str(fitness_list))
            if min_fitness < 0:
                fitness_list = [x - min_fitness for x in fitness_list]
                print("Fitness after normalization:\n" + str(fitness_list))
            fitness_list = self.__share_fitness__(population, fitness_list)
            print("Fitness after sharing:\n" + str(fitness_list))

            '''recalculate SAW array'''
            self.saw = self.__stepwise_adaption_of_weights__(self.saw)
            print("\nSAW: ", self.saw, "\n")

            '''every 10 generations check champion against ultimate'''
            if generation % 10 == 0:
                print("The best fitness was in generation %d and had a fitness of %.3f" % (
                self.best_gen, self.highest_fitness))

                total_individual_gain = 0
                total_wins = 0
                for i in range(5):
                    print("CHAMPION OF GENERATION %d, RUN %d OF %d \n" %(generation, (i+1), 5))
                    individual_gain, wins = self.__run_best_against_all__()
                    total_individual_gain += individual_gain
                    total_wins += wins

                average_ig = total_individual_gain / 5
                average_wins = total_wins / 5
                print("CHAMPION OF GENERATION %d, IG: %d\n" %(generation, average_ig))
                self.__compare_to_ultimate__(average_ig, average_wins, self.best_individual[0])
                self.logger.log_individual(average_ig)

            '''create next gen'''
            parents = self.selector.tournament_percentage(population, fitness_list)
            survivors = self.selector.select_best_percentage(population, fitness_list, BEST_SURVIVOR_PERCENTAGE)

            '''create children'''
            children = self.recombinator.blend(parents, NPOP-len(survivors))
            #children, self.dev, self.rot = self.mutator.correlated_mutation(children, self.dev, self.rot)
            children, self.dev = self.mutator.uncorrelated_mutation_n_step_size(children, self.dev)

            '''combine survivors and children'''
            population = np.append(children, survivors, axis=0)


os.environ["SDL_VIDEODRIVER"] = "dummy"

group1 = Generalist3([1,2,3,4,5,6,7,8])
group1.run()
