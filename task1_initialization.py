import numpy as np

DOM_L = -1 #duplicate constant in initialization
DOM_U = 1 #duplicate constant in initialization


class Initialization:

    def uniform_initialization(self, n_pop, n_vars):
        pop = np.random.uniform(DOM_L, DOM_U, (n_pop, n_vars))
        return pop
