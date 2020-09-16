import numpy as np

class Initialization:
    def __init__(self,  dom_l, dom_u):
        self.dom_l = dom_l
        self.dom_u = dom_u

    def uniform_initialization(self, n_pop, n_vars):
        pop = np.random.uniform(self.dom_l, self.dom_u, (n_pop, n_vars))
        return pop
