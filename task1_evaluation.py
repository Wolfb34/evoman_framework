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
        for enemy in self.enemies:
            self.env.update_parameter('enemies', [enemy])
            for i in range(length):
                fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
                fitness_results[i] += fitness

        return fitness_results

    def simple_generalist_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            print("\n INDIVIDUAL %d OF %d" %(i+1, length))
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = np.sum(fitness)

        return fitness_results

    def sharing_generalist_eval(self, pop):
        length = pop.shape[0]
        fitness_results = np.empty(length)
        for i in range(length):
            print("\n INDIVIDUAL %d OF %d" %(i+1, length))
            fitness, player_life, enemy_life, game_run_time = self.env.play(pcont=pop[i])
            fitness_results[i] = np.sum(fitness)

        return fitness_results
