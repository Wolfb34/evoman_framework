import os
import sys
import numpy as np
from task1_selection import Selection
from task1_mutation import Mutation
from task1_recombination import Recombination
from task1_initialization import Initialization
from task1_evaluation import Evaluation
from task1_constants import *
from task1_logger import Logger
from demo_controller import player_controller
sys.path.insert(0, 'evoman')
from environment import Environment

experiment_name = 'task1_specialist1_enemy_{}'.format(ENEMY)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


env = Environment(experiment_name=experiment_name,level=2, enemies=ENEMY,
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  speed="fastest")
N_VARS = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

init = Initialization(DOM_L, DOM_U)
evaluator = Evaluation(env)
selector = Selection()
logger = Logger(experiment_name)
recombinator = Recombination()
mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)

population = init.uniform_initialization(NPOP, N_VARS)
best = None
for i in range(NGEN):
    print(i)
    fitness_list = evaluator.simple_eval(population)
    logger.log_results(fitness_list)
    parents = selector.select_best_percentage(population, fitness_list)
    ind = np.argmax(fitness_list)
    if best is None:
        best = (fitness_list[ind], population[ind])
    elif fitness_list[ind] > best[0]:
        best = (fitness_list[ind], population[ind])
    population = recombinator.simple(parents, NPOP)
    population = mutator.nonuniform_mutation(population)

ig_list = []
for i in range(5):
    fitness, p_health, e_health, time = env.play(pcont=np.array(best[1]))
    ig_list.append(p_health - e_health)
logger.log_individual(np.average(ig_list))
