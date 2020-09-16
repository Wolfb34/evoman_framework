################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import os
from task1_selection import Selection
from task1_mutation import Mutation
from task1_recombination import Recombination
from task1_initialization import Initialization
from task1_evaluation import Evaluation

experiment_name = 'task1_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

N_HIDDEN_NEURONS = 10
NPOP = 10
NGEN = 5

init = Initialization()
evaluator = Evaluation()
selector = Selection()
recombinator = Recombination()
mutator = Mutation()

population = init.uniform_initialization(NPOP, N_HIDDEN_NEURONS)
print(population)
for i in range(NGEN):
    fitness_list = evaluator.simple_eval(population)
    parents = selector.select_best_percentage(population, fitness_list)
    population = recombinator.simple(parents, NPOP)
    population = mutator.nonuniform_mutation(population)
