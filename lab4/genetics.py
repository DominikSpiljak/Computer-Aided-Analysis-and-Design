import numpy as np


class TargetFunction:
    """Function class used for optimization, counts calls and saves calculated results
    """

    def __init__(self, f):
        self.f = f
        self.results = {}
        self.no_calls = 0

    def __call__(self, *args):
        """Method that overrides and handles calling object as function

        Returns:
            int: function value for given parameters
        """
        self.no_calls += 1

        if args in self.results:
            return self.results[args]
        else:
            result = self.f(*args)
            self.results[args] = result
            return result

    def wrap(self, wrapper):
        def wrap_func_1d(arg):
            return self.__call__(*wrapper(arg))
        return TargetFunction(wrap_func_1d)

    def reset(self):
        self.no_calls = 0


class Individual:

    def __init__(self, value, fitness_func, binary_representation=True, limits=[0, 1]):
        self.binary_representation = binary_representation
        self.lower_bound, self.upper_bound = limits
        self.value = value
        self.fitness_func = fitness_func

        if binary_representation:
            self.no_bits = self.value.shape[1]

        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = self.fitness_func(*self.decode_value())

    def decode_value(self):
        if self.binary_representation:
            float_vals = np.array([sum([val[i] * 2 ** (len(val) - 1 - i)
                                        for i in range(len(val))]) for val in self.value])
            return self.lower_bound + float_vals / (2 ** self.no_bits - 1) * (self.upper_bound - self.lower_bound)
        else:
            return self.value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        value = self.value
        if self.binary_representation:
            value = self.decode_value()
        return "Values = {}, Fitness = {}".format(value, self.fitness)


class GeneticAlgorithm:

    def __init__(self, population_generation, num_func_calls, func, selection, combination, mutation, solution, verbose=True):
        self.population_generation = population_generation
        self.num_func_calls = num_func_calls
        self.func = func
        self.selection = selection
        self.combination = combination
        self.mutation = mutation
        self.solution = solution
        self.verbose = verbose

    def evolution(self):
        # Template method
        population = self.population_generation()
        min_fitness = self.solution(population).fitness
        i = 0
        while True:
            population, comb_population = self.selection(population)
            population.extend(self.mutation(self.combination(comb_population)))
            if self.solution(population).fitness < min_fitness:
                min_fitness = self.solution(population).fitness
                if self.verbose:
                    print("Found new best, iteration = {}, {}".format(
                        i, self.solution(population)))
                if abs(min_fitness) < 1e-6:
                    if self.verbose:
                        print()
                    print(
                        "Found a solution with absolute value less than 1e-6, terminating.")
                    return self.solution(population)
            i += 1
            if self.func.no_calls > self.num_func_calls:
                print(
                    'Number of function evaluations is larger than the limit, terminating')
                self.func.reset()
                return self.solution(population)

            if i % 100 == 0 and self.verbose:
                print('Iteration: {}, Current number of function evaluations: {}'.format(
                    i, self.func.no_calls))


def generate_population(population_size, func, n_vars, binary_representation=True, limits=[0, 1], precision=1):
    def population_generation():
        population = []
        lower_bound, upper_bound = limits
        if binary_representation:
            num_bits = int(
                np.ceil(np.log2(10 ** precision * (upper_bound - lower_bound))))
            for _ in range(population_size):
                population.append(Individual(
                    value=np.random.randint(0, 2, size=(n_vars, num_bits)),
                    fitness_func=func, binary_representation=binary_representation, limits=limits))

        else:
            for _ in range(population_size):
                population.append(
                    Individual(
                        value=(lower_bound + np.random.rand(n_vars)
                               * (upper_bound - lower_bound)),
                        fitness_func=func, binary_representation=binary_representation, limits=limits))

        return population

    return population_generation


def roulette_selection(elitism=True, no_elites=1):
    def choose_index(proportions):
        maxi = proportions[-1]
        rand = np.random.rand() * maxi
        i = 0
        while proportions[i] < rand:
            i += 1
        return i - 1

    def get_mapping(value, worst, best, lower_bound, upper_bound):
        return lower_bound + (upper_bound - lower_bound) * ((value - worst) / (best - worst))

    def selection(population):
        new_population = []
        comb_population = []

        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)

        min_fitness = sorted_population[0].fitness
        max_fitness = sorted_population[-1].fitness

        if elitism:
            new_population.extend(sorted_population[:no_elites])
            no_combs = len(population) - no_elites
        else:
            no_combs = len(population)

        # Proportions calculation
        proportions = [get_mapping(
            population[0].fitness, max_fitness, min_fitness, 0, 1)]
        for individual in population[1:]:
            proportions.append(
                proportions[-1] + get_mapping(individual.fitness, max_fitness, min_fitness, 0, 1))

        for _ in range(no_combs):
            comb_population.append(
                [population[choose_index(proportions)], population[choose_index(proportions)]])

        return new_population, comb_population

    return selection


def one_point_cross_binary():
    def one_point_cross_function(comb_population):
        children = []
        for pair in comb_population:
            child_vals = []
            for j in range(pair[0].value.shape[0]):
                idx = np.random.randint(0, pair[0].value.shape[1])
                child_vals.append(
                    list(pair[0].value[j][:idx]) + list(pair[1].value[j][idx:]))
            children.append(Individual(
                np.array(child_vals), fitness_func=pair[0].fitness_func,
                limits=[pair[0].lower_bound, pair[0].upper_bound],
                binary_representation=pair[0].binary_representation))
        return children

    return one_point_cross_function


def uniform_cross_binary():
    def uniform_cross_function(comb_population):
        children = []
        for pair in comb_population:
            child_vals = []
            for i in range(pair[0].value.shape[0]):
                single_val = []
                for j in range(pair[0].value.shape[1]):
                    parent = int(np.random.rand() > .5)
                    single_val.append(pair[parent].value[i][j])
                child_vals.append(single_val)
            children.append(Individual(
                np.array(child_vals), fitness_func=pair[0].fitness_func,
                limits=[pair[0].lower_bound, pair[0].upper_bound],
                binary_representation=pair[0].binary_representation))
        return children

    return uniform_cross_function


def arithmetic_cross_float():
    def arithmetic_cross_function(comb_population):
        children = []
        for pair in comb_population:
            child_vals = []
            for i in range(pair[0].value.shape[0]):
                rand = np.random.rand()
                child_vals.append(
                    pair[0].value[i] * rand + pair[1].value[i] * (1 - rand))

            children.append(Individual(
                np.array(child_vals), fitness_func=pair[0].fitness_func,
                limits=[pair[0].lower_bound, pair[0].upper_bound],
                binary_representation=pair[0].binary_representation))
        return children

    return arithmetic_cross_function


def heuristic_cross_float():
    def heuristic_cross_function(comb_population):
        children = []
        for pair in comb_population:
            child_vals = []
            for i in range(pair[0].value.shape[0]):
                rand = np.random.rand()
                child_vals.append(
                    rand * (pair[1].value[i] - pair[0].value[i]) + pair[1].value[i])
            children.append(Individual(
                np.array(child_vals), fitness_func=pair[0].fitness_func,
                limits=[pair[0].lower_bound, pair[0].upper_bound],
                binary_representation=pair[0].binary_representation))
        return children

    return heuristic_cross_function


def solution():
    def solution_func(population):
        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)
        return sorted_population[0]
    return solution_func


def mutation(mutation_probabilty, binary_representation=True, limits=[0, 1]):
    def mutation_func(children):
        lower_bound, upper_bound = limits

        for individual in children:
            for j in range(individual.value.shape[0]):
                rand = np.random.rand()
                if rand < mutation_probabilty:
                    if binary_representation:
                        idx = np.random.randint(0, individual.value.shape[1])
                        individual.value[j][idx] = int(
                            not individual.value[j][idx])
                    else:
                        rand_mutate = lower_bound + np.random.rand() * (upper_bound - lower_bound)
                        individual.value[j] = rand_mutate

            individual.calculate_fitness()
        return children

    return mutation_func


def print_task_num(num):
    print()
    print('#' * 40, 'zadatak ', num, '#' * 40)
    print()


def main():
    def f(x, y):
        return 0.5 + (np.sin(np.sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

    tf = TargetFunction(f)
    binary_representation = False
    limits = [-50, 150]
    genetic = GeneticAlgorithm(
        population_generation=generate_population(population_size=1000, func=tf, n_vars=2,
                                                  binary_representation=binary_representation,
                                                  limits=limits, precision=3),
        num_func_calls=500000,
        func=tf,
        selection=roulette_selection(
            elitism=True, no_elites=500),
        combination=heuristic_cross_float(),
        mutation=mutation(
            0.1, binary_representation=binary_representation, limits=limits),
        solution=solution(),
        verbose=True)

    best = genetic.evolution()
    print('Best: {}'.format(best))


if __name__ == "__main__":
    main()
