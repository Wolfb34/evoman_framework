import numpy as np
from task1_initialization import Initialization
from task1_constants import *


class Mutation:
    def __init__(self, min_dev, rotation, stdev, dom_l, dom_u):
        self.min_dev = min_dev # picked at random, do research into what is best.
        self.rotation = rotation
        self.stdev = stdev # picked at random
        self.dom_l = dom_l # duplicate constant in initialization
        self.dom_u = dom_u # duplicate constant in initialization

    #applies mutation to every gene by some amount drawn from a normal distribution
    def nonuniform_mutation(self, pop):
        n_pop = len(pop)
        n_vars = len(pop[0])
        new_pop = np.copy(pop)

        for i in range(n_pop):
            for j in range(n_vars):
                mutation_amount = np.random.normal(scale=self.stdev)
                gene = pop[i, j]
                new_gene = np.clip(gene + mutation_amount, self.dom_l, self.dom_u)
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
                if new_dev[i, j] < self.min_dev:
                    new_dev[i, j] = self.min_dev

                mutation_amount = np.random.normal(scale=new_dev[i, j])
                gene = pop[i, j]
                new_gene = np.clip(gene + mutation_amount, self.dom_l, self.dom_u)
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
                rotation_mutation_amount = np.random.normal(scale=self.rotation)
                new_rot[i, j] = rot[i, j] + rotation_mutation_amount
                if abs(new_rot[i, j]) > np.pi:
                    new_rot[i, j] = new_rot[i, j] - (2*np.pi*np.sign(new_rot[i, j]))

            #mutate standard deviations
            individual_mutation = np.random.normal(scale=tau_prime)
            for j in range(n_vars):
                gene_mutation = np.random.normal(scale=tau)
                new_dev[i, j] = dev[i, j] * (np.e**(individual_mutation + gene_mutation))
                if new_dev[i, j] < self.min_dev:
                    new_dev[i, j] = self.min_dev

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
            new_pop[i] = np.clip(pop[i] + mutation_amounts, self.dom_l, self.dom_u)

        return new_pop, new_dev, new_rot

if __name__ == "__main__":
    m = Mutation(MIN_DEV, ROTATION_MUTATION, STANDARD_DEVIATION, DOM_L, DOM_U)
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
