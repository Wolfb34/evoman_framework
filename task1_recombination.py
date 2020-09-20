import numpy as np
import itertools

class Recombination:


    """Choose pairs of parents to create offspring until the population is filled.
       To create children, choose a cut-off point and use the first parent to the left,
       second parent to the right for first child. Reverse for second. Returns
       the new population.
       parents - all the parents that were selected to reproduce
       npop - the total population"""
    def simple(self, parents, npop):
        nvar = len(parents[0])
        next_gen = np.zeros([npop, nvar])
        nums = [i for i in range(len(parents))]
        pairs = [pair for pair in itertools.combinations(nums, 2)]
        if len(nums) < 2:
            print("Need at least 2 parents for recombination to be possible")
            return None

        for i in range(0, npop, 2):
            index = np.random.choice(nums[:-1])
            donor1 = parents[pairs[index][0]]
            donor2 = parents[pairs[index][1]]

            cut_point = np.random.randint(1, nvar - 1)

            child1 = []
            child2 = []
            for j in range(nvar):
                if j < cut_point:
                    child1.append(donor1[j])
                    child2.append(donor2[j])
                else:
                    child1.append(donor2[j])
                    child2.append(donor1[j])

            next_gen[i] = child1
            if i + 1 < npop:
                next_gen[i + 1] = child2
        return next_gen

    """Choose pairs of parents to create offspring until the population is filled.
       To create children, find the differences between every entry of the parents,
       then pick an alpha between 0 and 1 and create children by choosing elements
       between min_i - alpha * diff_i and min_i + alpha * diff_i. Returns
       the new population.
       parents - all the parents that were selected to reproduce
       npop - the total population"""
    def blend(self, parents, npop):
        nvar = len(parents[0])
        next_gen = np.zeros([npop, nvar])
        nums = [i for i in range(len(parents))]
        pairs = [pair for pair in itertools.combinations(nums, 2)]
        if len(nums) < 2:
            print("Need at least 2 parents for recombination to be possible")
            return None

        for i in range(0, npop, 2):
            index = np.random.choice(nums[:-1])

            donor1 = parents[pairs[index][0]]
            donor2 = parents[pairs[index][1]]

            minarray = list(map(min, donor1, donor2))
            maxarray = list(map(max, donor1, donor2))

            diff = list(map(lambda x, y: x - y, maxarray, minarray))
            alpha = np.random.uniform(0, 1)

            low = [minarray[i] - alpha * diff[i] for i in range(nvar)]
            high = [maxarray[i] + alpha * diff[i] for i in range(nvar)]

            child1 = np.random.uniform(low, high)
            next_gen[i] = child1
            if i + 1 < npop:
                child2 = np.random.uniform(low, high)
                next_gen[i + 1] = child2
        return next_gen

if __name__ == "__main__":
    NVAR = 4
    NPOP = 4
    parent_array = np.random.uniform(0, 1, (NPOP, NVAR))
    print(parent_array)
    recon = Recombination()
    new_gen = recon.blend(parent_array, 6)
    print(new_gen)
