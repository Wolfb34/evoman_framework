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




experiment_name = '(task2_specialist, enemy: {})'.format(ENEMY)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(level=2,
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemy=ENEMY,
                  speed="fastest",
                  enemies=[1])


n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
rot_size = int((n_vars * (n_vars - 1)) / 2)
dev = np.random.uniform(0, INIT_SD, (NPOP, n_vars))
rot = np.random.uniform(-np.pi, np.pi, (NPOP, rot_size))


init = Initialization(DOM_L, DOM_U)
evaluator = Evaluation(env)
selector = Selection()
logger = Logger(experiment_name)
recombinator = Recombination()
mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)

'''
Changes with regards to specialist1:

* Specialist 2 uses tournament selection (instead of selecting the best)
* Specialist 2 uses blend recombination (instead of simple)
* Specialist 2 uses correlated mutation (instead of nonuniform)

'''

population = init.uniform_initialization(NPOP, n_vars)
fitness_list = []
print(population)
for i in range(1,NGEN+1):
    print("EVALUATION GENERATION %d\n" %i)
    fitness_list = evaluator.simple_eval(population)
    logger.log_results(fitness_list)

    if i != NGEN:
        parents = selector.tournament_percentage(population, fitness_list)
        population = recombinator.blend(parents, NPOP)
        population, dev, rot = mutator.correlated_mutation(population, dev, rot)
        #population, dev = mutator.uncorrelated_mutation_n_step_size(population, dev)



# Select best from the last generation
champion = selector.select_best_n(population,fitness_list, 1)
individual_gain = []

print("The champion is: %s" %champion)
for run in range(5):
    fitness_list = evaluator.simple_eval(champion)

    fitness, player_life, enemy_life, game_run_time = env.play(pcont=champion)
    individual_gain.append(player_life-enemy_life)

average_ig = sum(individual_gain)/len(individual_gain)
logger.log_individual(average_ig)

