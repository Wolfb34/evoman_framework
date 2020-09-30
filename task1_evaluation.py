import numpy as np

class Evaluation:
    def __init__(self, env):
        self.env = env

    #makes use of the fitness function without any changes
    def simple_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = fitness

        return fitness_results

    def multiplemode_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = np.sum(fitness)

        return fitness_results
