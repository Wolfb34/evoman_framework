import numpy as np

class Evaluation:
    def __init__(self, env, share_size):
        self.env = env
        self.share_size = share_size

    #makes use of the fitness function without any changes
    def simple_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = fitness

        return fitness_results
        
    def __share_of__(self, ind1, ind2):
        dist = np.linalg.norm(ind1 - ind2)
        if dist > self.share_size:
            return 0
        return 1 - dist / self.share_size

    def share_fitness(self, pop, fitness):
        new_fitness = []
        length = len(pop)
        for i in range(length):
            divisor = sum([self.__share_of__(pop[i], pop[j]) for j in range(length)])
            new_fitness.append(fitness[i] / divisor)
        return np.array(new_fitness)

    def simple_generalist_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = np.sum(fitness)

        return fitness_results
