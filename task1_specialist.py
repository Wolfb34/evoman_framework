################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import os
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from task1_selection import Selection
from task1_mutation import Mutation
from task1_recombination import Recombination
from task1_initialization import Initialization
from task1_evaluation import Evaluation
from task1_constants import *
from demo_controller import player_controller
from environment import Environment

experiment_name = 'task1_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


env = Environment(level=2,
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  speed="fastest")
init = Initialization()
evaluator = Evaluation(env)
selector = Selection()
recombinator = Recombination()
mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)

n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5


population = init.uniform_initialization(NPOP, n_vars)
for i in range(NGEN):
    print(i)
    fitness_list = evaluator.simple_eval(population)
    parents = selector.select_best_percentage(population, fitness_list)
    population = recombinator.simple(parents, NPOP)
    population = mutator.nonuniform_mutation(population)
