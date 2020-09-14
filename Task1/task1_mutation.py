import numpy as np
from Task1.task1_initialization import Initialization
MIN_DEV = 0.0001 #picked at random, do research into what is best.
ROTATION_MUTATION = np.radians(5)
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
        n_rot = int((n_vars*(n_vars-1))/2)

        new_pop = np.copy(pop)
        new_dev = np.copy(dev)
        new_rot = np.copy(rot)

        tau = 1/np.sqrt(2*n_vars)
        tau_prime = 1/np.sqrt(2*np.sqrt(n_vars))

        for i in range(n_pop):

            #mutate rotations
            for j in range(n_rot):
                rotation_mutation_amount = np.random.normal(scale=ROTATION_MUTATION)
                new_rot[i, j] = rot[i, j] + rotation_mutation_amount
                if abs(new_rot[i, j]) > np.pi:
                    new_rot[i, j] = new_rot[i, j] - (2*np.pi*np.sign(new_rot[i, j]))

            #mutate standard deviations
            individual_mutation = np.random.normal(scale=tau_prime)
            for j in range(n_vars):
                gene_mutation = np.random.normal(scale=tau)
                new_dev[i, j] = dev[i, j] * (np.e**(individual_mutation + gene_mutation))
                if new_dev[i, j] < MIN_DEV:
                    new_dev[i, j] = MIN_DEV

            #create covariance matrix
            covariance_matrix = np.zeros((n_vars, n_vars))
            rotations_added = 0
            for j in range(n_vars):
                for k in range(n_vars):
                    if j == k:
                        covariance_matrix[j, k] = new_dev[i, j]**2
                    elif j < k: #there is a correlation
                        covariance_matrix[j, k] = 0.5*((new_dev[i, j]**2) - (new_dev[i, k]**2))\
                                                  *np.tan(2*new_rot[i, rotations_added])
                        rotations_added += 1
                    else: #there is not a correlation
                        covariance_matrix[j, k] = 0

            zeros_array = np.zeros(n_vars)
            mutation_amounts = np.random.multivariate_normal(mean=zeros_array, cov=covariance_matrix)
            new_pop[i] = np.clip(pop[i] + mutation_amounts, DOM_L, DOM_U)

        return new_pop, new_dev, new_rot

m = Mutation()
i = Initialization()

size = 10
rot_size = int((size*(size-1))/2)
pop = i.uniform_initialization(size, size)
dev = np.random.uniform(0, 0.0001, (size, size))
rot = np.random.uniform(-np.pi, np.pi, (size, rot_size))

for i in range(3):
    print(pop)
    print(dev)
    print(rot)
    print()
    pop, dev, rot = m.correlated_mutation(pop, dev, rot)
    print(pop)
    print(dev)
    print(rot)
