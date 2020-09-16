import numpy, math, random

class Selection:

    '''
    This method selects the best fraction of the population based on their fitness.
    *Arguments:
        - population: a list of individuals
        - fitness: a list of the fitness values corresponding to the population
        - fraction: the percentage of the population you want to select as parents. This value must be 0 <= value <= 1.
          The default fraction is 0.25
    * Return:
        - A list of individuals
    '''
    def select_best_percentage(self, population, fitness, fraction=0.25):
        if not(0 <= fraction <= 1):
            print("Error: invalid fraction %.2f" %fraction)
        return self.select_best_n(population, fitness, int(len(population) * fraction))

    '''
    This method selects the best n individuals from the population based on their fitness.
    *Arguments:
        - population: a list of individuals
        - fitness: a list of the fitness values corresponding to the population
        - n: the number of parents you want to select
    * Return:
        - A list of n individuals
    '''
    def select_best_n(self, population, fitness, n):
        if not self.__check_match__(population, fitness):
            return population

        parents = []

        fitness_indices_sorted = (list(numpy.argsort(fitness)))
        for i in fitness_indices_sorted[-n::]:
            parents.append(population[i])

        return parents

    '''
     This method selects n individuals from the population by applying the tournament-method based on their fitness.
     *Arguments:
         - population: a list of individuals
         - fitness: a list of the fitness values corresponding to the population
         - n: the number of parents you want to select
     * Return:
         - A list of n individuals
     '''
    def tournament_n(self, population, fitness, n):
        if not self.__check_match__(population, fitness):
            return population

        tournament_size = len(population) / n
        parents = []

        for battle in range(n):
            lower = math.ceil(tournament_size * battle)
            upper = min(math.ceil(tournament_size * (battle+1)), len(population))

            parents.append(self.select_best_n(population[lower:upper], fitness[lower:upper], 1)[0])

        return parents

    '''
    This method selects a fraction of the population by applying the tournament-method based on their fitness.
    *Arguments:
        - population: a list of individuals
        - fitness: a list of the fitness values corresponding to the population
        - fraction: the percentage of the population you want to select as parents. This value must be 0 <= value <= 1.
          The default fraction is 0.25.
    * Return:
        - A list of individuals
    '''
    def tournament_percentage(self, population, fitness, fraction=0.25):
        if not(0 <= fraction <= 1):
            print("Error: invalid fraction %.2f" %fraction)
        return self.tournament_n(population, fitness, int(len(population)*fraction))

    def __check_match__(self, population, fitness):
        if len(population) != len(fitness):
            print("Error: The fitness does not seem to match the population\n")
            return False
        return True

if __name__ == "__main__":
    '''Test code'''
    NUMBER = 26
    fit = list(range(0,NUMBER))
    random.shuffle(fit)

    pop = []
    for c in range(ord('a'), ord('a')+NUMBER):
        pop.append(chr(c))

    print("Individuals and their fitness:")
    for i in range(NUMBER):
        print("%s: %d" %(pop[i], fit[i]), end='\t')
    print("\n")

    s = Selection()

    print("Parents best_4:          %s" % s.select_best_n(pop, fit, 4))
    print("Parents tournament_4:    %s" %s.tournament_n(pop, fit,4))
    print()
    print("Parents best_35%%:        %s" % s.select_best_percentage(pop, fit, 0.35))
    print("Parents tournament_35%%:  %s" %s.tournament_percentage(pop, fit, 0.35))
