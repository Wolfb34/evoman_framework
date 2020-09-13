import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

N_HIDDEN_NEURONS = 10


class Evaluation:

    #makes use of the fitness function without any changes
    def simple_eval(self, pop):
        env = Environment(level=2,
                          player_controller=player_controller(N_HIDDEN_NEURONS),
                          speed="fastest")
        fitness_results = np.empty(pop.len())

        for i in pop:
            fitness, player_life, enemy_life, game_run_time = env.play(pcont=i)
            fitness_results[i] = fitness

        return fitness_results
