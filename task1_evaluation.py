import numpy as np

class Evaluation:
    def __init__(self, env, enemies, share_size):
        self.env = env
        self.enemies = enemies
        self.share_size = share_size

    #makes use of the fitness function without any changes
    def simple_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            fitness, _, _, _ = self.env.play(pcont=pop[i])
            fitness_results[i] = fitness

        return fitness_results

    def simple_generalist_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            print("\n INDIVIDUAL %d OF %d" %(i+1, length))
            fitness_individual = 0

            for enemy in self.enemies:
                self.env.update_parameter('enemies', [enemy])
                fitness, _, _, _ = self.env.play(pcont=pop[i])
                fitness_individual += fitness

            fitness_results[i] = np.sum(fitness_individual)

        return fitness_results

    def sharing_generalist_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            print("\n INDIVIDUAL %d OF %d" % (i + 1, length))
            fitness_individual = 0

            for enemy in self.enemies:
                self.env.update_parameter('enemies', [enemy])
                fitness, _, _, _ = self.env.play(pcont=pop[i])
                fitness_individual += fitness

            fitness_results[i] = np.sum(fitness_individual)

        return fitness_results
