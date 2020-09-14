import numpy as np

MIN_DEV = 0.0001 #picked at random, do research into what is best.
STANDARD_DEVIATION = 0.01 #picked at random
DOM_L = -1 #duplicate constant in initialization
DOM_U = 1 #duplicate constant in initialization


class Mutation:

    #applies mutation to every gene by some amount drawn from a normal distribution
    def nonuniform_mutation(self, pop):
        n_pop = len(pop)
        n_vars = len(pop[0])
        new_pop = np.copy(pop)

        for i in range(n_pop):
            for j in range(n_vars):
                mutation_amount = np.random.normal(scale=STANDARD_DEVIATION)
                gene = pop[i, j]
                new_gene = np.clip(gene + mutation_amount, DOM_L, DOM_U)
                new_pop[i, j] = new_gene

        return new_pop

    #input population and standard deviations
    def uncorrelated_mutation_n_step_size(self, pop, dev):
        n_pop = len(pop)
        n_vars = len(pop[0])
        new_pop = np.copy(pop)
        new_dev = np.copy(dev)

        tau = 1/np.sqrt(2*n_vars)
        tau_prime = 1/np.sqrt(2*np.sqrt(n_vars))

        for i in range(n_pop):
            individual_mutation = np.random.normal(scale=tau_prime)
            for j in range(n_vars):
                gene_mutation = np.random.normal(scale=tau)
                new_dev[i, j] = dev[i, j] * (np.e**(individual_mutation + gene_mutation))
                if new_dev[i, j] < MIN_DEV:
                    new_dev[i, j] = MIN_DEV

                mutation_amount = np.random.normal(scale=new_dev[i, j])
                gene = pop[i, j]
                new_gene = np.clip(gene + mutation_amount, DOM_L, DOM_U)
                new_pop[i, j] = new_gene

        return new_pop, new_dev

    def correlated_mutation(self, pop, dev, rot):
        n_pop = len(pop)
        n_vars = len(pop[0])
        n_rot = (n_vars*(n_vars-1))/2

        new_pop = np.copy(pop)
        new_dev = np.copy(dev)
        new_rot = np.copy(rot)

        for i in range(n_pop):
            #create covariance matrix
            covariance_matrix = np.zeros((n_vars, n_vars))
            

        return new_pop, new_dev
