import numpy as np

STANDARD_DEVIATION = 0.01 #picked at random
DOM_L = -1 #duplicate constant in initialization
DOM_U = 1 #duplicate constant in initialization


class Mutation:

    #applies mutation to every gene by some amount drawn from a normal distribution
    def nonuniform_mutation(self, pop):
        new_pop = np.copy(pop)
        for i in pop:
            for j in i:
                mutation_amount = np.random.normal(scale=STANDARD_DEVIATION)
                gene = pop[i, j]
                new_gene = np.clip(gene + mutation_amount, DOM_L, DOM_U)
                new_pop[i, j] = new_gene

        return new_pop

    def correlated_mutation(self, pop):
        new_pop = np.copy(pop)
        return new_pop

