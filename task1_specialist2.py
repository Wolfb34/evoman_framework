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
from task1_recombination import Recombination
from task1_initialization import Initialization
from task1_evaluation import Evaluation
from environment import Environment
from task1_constants import *



experiment_name = 'task1_specialist2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
			  	  speed="fastest",
				  enemymode="static",
				  level=2)


rot_size = int((NPOP * (NPOP - 1)) / 2)
dev = np.random.uniform(0, 0.01, (NPOP, NPOP))
rot = np.random.uniform(-np.pi, np.pi, (NPOP, rot_size))


init = Initialization(DOM_L, DOM_U)
evaluator = Evaluation(env)
selector = Selection()
recombinator = Recombination()
mutator = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)


'''
Changes with regards to specialist1:

* Specialist 2 uses tournament selection (instead of selecting the best)
* Specialist 2 uses blend recombination (instead of simple)
* Specialist 2 uses correlated mutation (instead of nonuniform)

'''

population = init.uniform_initialization(NPOP, N_HIDDEN_NEURONS)
print(population)
for i in range(NGEN):
    print("EVALUATION GENERATION %d\n" %i)
    fitness_list = evaluator.simple_eval(population)
    parents = selector.tournament_percentage(population, fitness_list)

    population = recombinator.blend(parents, NPOP)

    population, dev, rot = mutator.correlated_mutation(population, dev, rot)
    #population, dev = mutator.uncorrelated_mutation_n_step_size(population, dev)
