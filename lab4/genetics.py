import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)


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
        return np.all(self.value == other.value)

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
                    self.func.reset()
                    return self.solution(population)
            i += 1
            if self.func.no_calls > self.num_func_calls:
                print(
                    'Number of function evaluations is larger than the limit, terminating')
                self.func.reset()
                return self.solution(population)

            if self.func.no_calls % 1000 == 0 and self.verbose:
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


def tournament_selection(k=3):
    def selection(population):
        selected = np.random.choice(population, k, replace=False)
        selected_sorted = sorted(selected, key=lambda x: x.fitness)
        comb_population = [[selected_sorted[0], selected_sorted[1]]]
        population.remove(selected_sorted[-1])

        return population, comb_population

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
            pair = sorted(pair, key=lambda x: -x.fitness)
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


def zadatak_1(f1, f3, f6, f7, no_experiments, verbose, plot_name, population_size=100, num_func_calls=1e5, show_plot=True):
    ######################################## zadatak  1 ########################################

    print_task_num(1)
    binary_solutions = {0: [], 1: [], 2: [], 3: []}
    float_solutions = {0: [], 1: [], 2: [], 3: []}
    binary_no_elites = int(population_size / 3)
    floating_no_elites = int(population_size / 2)
    limits = [-50, 150]
    for experiment in range(no_experiments):
        print("Experiment {}: ".format(experiment))
        for i, f in enumerate([f1, f3, f6, f7]):
            print('Pokrecem rijesavanje za funkciju {}'.format(i + 1))
            print('Binarni prikaz')
            binary_representation = True
            genetic = GeneticAlgorithm(
                population_generation=generate_population(population_size=population_size, func=f, n_vars=2,
                                                          binary_representation=binary_representation,
                                                          limits=limits, precision=5),
                num_func_calls=num_func_calls,
                func=f,
                selection=roulette_selection(
                    elitism=True, no_elites=binary_no_elites),
                combination=uniform_cross_binary(),
                mutation=mutation(
                    0.8, binary_representation=binary_representation, limits=limits),
                solution=solution(),
                verbose=verbose)

            best_binary = genetic.evolution()
            print('Najbolji: {}'.format(best_binary))
            binary_solutions[i].append(best_binary.fitness)

            print("--------")

            print('Prikaz s pomicnom tockom')
            binary_representation = False
            genetic = GeneticAlgorithm(
                population_generation=generate_population(population_size=population_size, func=f, n_vars=2,
                                                          binary_representation=binary_representation,
                                                          limits=limits, precision=3),
                num_func_calls=num_func_calls,
                func=f,
                selection=roulette_selection(
                    elitism=True, no_elites=floating_no_elites),
                combination=heuristic_cross_float(),
                mutation=mutation(
                    0.1, binary_representation=binary_representation, limits=limits),
                solution=solution(),
                verbose=verbose)

            best_float = genetic.evolution()
            print('Najbolji: {}'.format(best_float))
            float_solutions[i].append(best_float.fitness)

            print('\n########################################\n')
    if no_experiments > 1:
        functions = ['f1', 'f3', 'f6', 'f7']
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(20, 15)
        fig.suptitle('Rezultati nakon {} eksperimenata za svaku funkciju\n\
                      Parametri za binarni prikaz: population_size={}, precision=5, no_elites={}, cross_alg=uniform_cross_binary, mutation_prob=0.8\n\
                      Parametri za prikaz s pomicnom tockom: population_size={}, no_elites={}, cross_alg=heuristic_cross_float, mutation_prob=0.1'.format(no_experiments, population_size, binary_no_elites, population_size, floating_no_elites))

        for i, f in enumerate(functions):
            axes_coords = [0, i] if i < 2 else [1, i - 2]
            axes[axes_coords[0]][axes_coords[1]].set_title(
                'Funkcija {}'.format(f))
            axes[axes_coords[0]][axes_coords[1]].boxplot([binary_solutions[i], float_solutions[i]], labels=[
                'Binarni prikaz', 'Prikaz s pomicnom tockom'])
        plt.savefig(plot_name)
        if show_plot:
            plt.show()


def zadatak_2(f6, f7, no_experiments, verbose, plot_name, population_size=100, num_func_calls=1e5, show_plot=True):
    ######################################## zadatak  2 ########################################

    print_task_num(2)

    binary_no_elites = int(population_size / 3)
    floating_no_elites = int(population_size / 2)

    binary_solutions = {0: {1: [], 3: [], 6: [], 10: []},
                        1: {1: [], 3: [], 6: [], 10: []}}
    float_solutions = {0: {1: [], 3: [], 6: [], 10: []},
                       1: {1: [], 3: [], 6: [], 10: []}}
    limits = [-50, 150]
    for experiment in range(no_experiments):
        print("Experiment {}: ".format(experiment))
        for i, n_vars in enumerate([1, 3, 6, 10]):
            for j, f in enumerate([f6, f7]):
                print(
                    'Pokrecem rijesavanje za funkciju {} i dimenziju {}'.format(j + 1, n_vars))
                print('Binarni prikaz')
                binary_representation = True
                genetic = GeneticAlgorithm(
                    population_generation=generate_population(population_size=population_size, func=f, n_vars=n_vars,
                                                              binary_representation=binary_representation,
                                                              limits=limits, precision=5),
                    num_func_calls=num_func_calls,
                    func=f,
                    selection=roulette_selection(
                        elitism=True, no_elites=binary_no_elites),
                    combination=uniform_cross_binary(),
                    mutation=mutation(
                        0.8, binary_representation=binary_representation, limits=limits),
                    solution=solution(),
                    verbose=verbose)

                best_binary = genetic.evolution()
                print('Najbolji: {}'.format(best_binary))
                binary_solutions[j][n_vars].append(best_binary.fitness)

                print("--------")

                print('Prikaz s pomicnom tockom')
                binary_representation = False
                genetic = GeneticAlgorithm(
                    population_generation=generate_population(population_size=population_size, func=f, n_vars=n_vars,
                                                              binary_representation=binary_representation,
                                                              limits=limits, precision=3),
                    num_func_calls=num_func_calls,
                    func=f,
                    selection=roulette_selection(
                        elitism=True, no_elites=floating_no_elites),
                    combination=heuristic_cross_float(),
                    mutation=mutation(
                        0.1, binary_representation=binary_representation, limits=limits),
                    solution=solution(),
                    verbose=verbose)

                best_float = genetic.evolution()
                print('Najbolji: {}'.format(best_float))
                float_solutions[j][n_vars].append(best_float.fitness)

                print('\n########################################\n')

    if no_experiments > 1:
        dimensions = [1, 3, 6, 10]
        functions = ['f6', 'f7']
        fig, axes = plt.subplots(2, 4)
        fig.set_size_inches(20, 15)
        fig.suptitle('Rezultati nakon {} eksperimenata za svaku funkciju\n\
                      Parametri za binarni prikaz: population_size={}, precision=5, no_elites={}, cross_alg=uniform_cross_binary, mutation_prob=0.8\n\
                      Parametri za prikaz s pomicnom tockom: population_size={}, no_elites={}, cross_alg=heuristic_cross_float, mutation_prob=0.1'.format(no_experiments, population_size, binary_no_elites, population_size, floating_no_elites))

        for i, dim in enumerate(dimensions):
            for j, f in enumerate(functions):
                axes[j][i].set_title(
                    'Funkcija {}, Dimenzija {}'.format(f, dim))
                axes[j][i].boxplot([binary_solutions[j][dim], float_solutions[j][dim]], labels=[
                                   'Binarni prikaz', 'Prikaz s pomicnom tockom'])
        plt.savefig(plot_name)
        if show_plot:
            plt.show()


def zadatak_3(f6, f7, no_experiments, verbose, plot_name, population_size=100, num_func_calls=1e5, show_plot=True):
    ######################################## zadatak  3 ########################################

    print_task_num(3)

    no_elites = int(population_size / 2)

    binary_solutions = {0: {3: [], 6: []},
                        1: {3: [], 6: []}}
    float_solutions = {0: {3: [], 6: []},
                       1: {3: [], 6: []}}
    limits = [-50, 150]
    for experiment in range(no_experiments):
        print("Experiment {}: ".format(experiment))
        for i, n_vars in enumerate([3, 6]):
            for j, f in enumerate([f6, f7]):
                print(
                    'Pokrecem rijesavanje za funkciju {} i dimenziju {}'.format(j + 1, n_vars))
                print('Binarni prikaz')
                binary_representation = True
                genetic = GeneticAlgorithm(
                    population_generation=generate_population(population_size=population_size, func=f, n_vars=n_vars,
                                                              binary_representation=binary_representation,
                                                              limits=limits, precision=4),
                    num_func_calls=num_func_calls,
                    func=f,
                    selection=roulette_selection(
                        elitism=True, no_elites=no_elites),
                    combination=uniform_cross_binary(),
                    mutation=mutation(
                        0.1, binary_representation=binary_representation, limits=limits),
                    solution=solution(),
                    verbose=verbose)

                best_binary = genetic.evolution()
                print('Najbolji: {}'.format(best_binary))
                binary_solutions[j][n_vars].append(best_binary.fitness)

                print("--------")

                print('Prikaz s pomicnom tockom')
                binary_representation = False
                genetic = GeneticAlgorithm(
                    population_generation=generate_population(population_size=population_size, func=f, n_vars=n_vars,
                                                              binary_representation=binary_representation,
                                                              limits=limits, precision=4),
                    num_func_calls=num_func_calls,
                    func=f,
                    selection=roulette_selection(
                        elitism=True, no_elites=no_elites),
                    combination=heuristic_cross_float(),
                    mutation=mutation(
                        0.1, binary_representation=binary_representation, limits=limits),
                    solution=solution(),
                    verbose=verbose)

                best_float = genetic.evolution()
                print('Najbolji: {}'.format(best_float))
                float_solutions[j][n_vars].append(best_float.fitness)

                print('\n########################################\n')

    if no_experiments > 1:
        dimensions = [3, 6]
        functions = ['f6', 'f7']
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(20, 15)
        fig.suptitle('Rezultati nakon {} eksperimenata za svaku funkciju\n\
                      Parametri za binarni prikaz: population_size={}, precision=4, no_elites={}, cross_alg=uniform_cross_binary, mutation_prob=0.4\n\
                      Parametri za prikaz s pomicnom tockom: population_size={}, no_elites={}, cross_alg=heuristic_cross_float, mutation_prob=0.4'.format(no_experiments, population_size, no_elites, population_size, no_elites))

        for i, dim in enumerate(dimensions):
            for j, f in enumerate(functions):
                axes[j][i].set_title(
                    'Funkcija {}, Dimenzija {}'.format(f, dim))
                axes[j][i].boxplot([binary_solutions[j][dim], float_solutions[j][dim]], labels=[
                                   'Binarni prikaz', 'Prikaz s pomicnom tockom'])
        plt.savefig(plot_name)
        if show_plot:
            plt.show()


def zadatak_4(f6, no_experiments, verbose, plot_name, num_func_calls=1e5, show_plot=True):
    ######################################## zadatak  4 ########################################

    print_task_num(4)

    population_sizes = [30, 50, 100, 200]
    no_elites_sizes = [int(population_size / 2)
                       for population_size in population_sizes]
    mutation_probs = [0.1, 0.3, 0.6, 0.9]

    float_solutions = {pop: {mut: [] for mut in mutation_probs}
                       for pop in population_sizes}
    limits = [-50, 150]
    for experiment in range(no_experiments):
        print("Experiment {}: ".format(experiment))
        for no_elites, population_size in zip(no_elites_sizes, population_sizes):
            for mutation_prob in mutation_probs:
                print(
                    'Pokrecem rijesavanje za velicinu populacije {} i vjerojatnost mutacije {}'.format(population_size, mutation_prob))

                print('Prikaz s pomicnom tockom')
                binary_representation = False
                genetic = GeneticAlgorithm(
                    population_generation=generate_population(population_size=population_size, func=f6, n_vars=2,
                                                              binary_representation=binary_representation,
                                                              limits=limits, precision=4),
                    num_func_calls=num_func_calls,
                    func=f6,
                    selection=roulette_selection(
                        elitism=True, no_elites=no_elites),
                    combination=heuristic_cross_float(),
                    mutation=mutation(
                        0.1, binary_representation=binary_representation, limits=limits),
                    solution=solution(),
                    verbose=verbose)

                best_float = genetic.evolution()
                print('Najbolji: {}'.format(best_float))
                float_solutions[population_size][mutation_prob].append(
                    best_float.fitness)

                print('\n########################################\n')

    if no_experiments > 1:
        fig, axes = plt.subplots(4, 4)
        fig.set_size_inches(20, 15)
        fig.tight_layout(pad=3.0)
        for i, population_size in enumerate(population_sizes):
            for j, mutation_prob in enumerate(mutation_probs):
                axes[j][i].set_title(
                    'Velicina populacije {}, Vjerojatnost mutacije {}'.format(population_size, mutation_prob), pad=15)
                axes[j][i].boxplot([float_solutions[population_size][mutation_prob]], labels=[
                                   'Prikaz s pomicnom tockom'])
        plt.savefig(plot_name)
        if show_plot:
            plt.show()


def zadatak_5(f6, no_experiments, verbose, plot_name, population_size=100, num_func_calls=1e5, show_plot=True):
    ######################################## zadatak  4 ########################################

    print_task_num(5)
    tournament_sizes = [3, 5, 15, 30]
    float_solutions = {t_size: [] for t_size in tournament_sizes}
    limits = [-50, 150]

    for experiment in range(no_experiments):
        print("Experiment {}: ".format(experiment))
        for t_size in tournament_sizes:
            print('Pokrecem rijesavanje za velicinu turnira {}'.format(
                t_size))

            print('Prikaz s pomicnom tockom')
            binary_representation = False
            genetic = GeneticAlgorithm(
                population_generation=generate_population(population_size=population_size, func=f6, n_vars=2,
                                                          binary_representation=binary_representation,
                                                          limits=limits, precision=4),
                num_func_calls=num_func_calls,
                func=f6,
                selection=tournament_selection(t_size),
                combination=heuristic_cross_float(),
                mutation=mutation(
                    0.4, binary_representation=binary_representation, limits=limits),
                solution=solution(),
                verbose=verbose)

            best_float = genetic.evolution()
            print('Najbolji: {}'.format(best_float))
            float_solutions[t_size].append(best_float.fitness)

            print('\n########################################\n')

    if no_experiments > 1:
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(12, 8)
        fig.suptitle('Rezultati nakon {} eksperimenata za svaku funkciju\n\
                      Parametri za prikaz s pomicnom tockom: population_size={}, tournament_sizes={}, cross_alg=heuristic_cross_float, mutation_prob=0.1'.format(no_experiments, population_size, tournament_sizes))
        for i, t_size in enumerate(tournament_sizes):
            axes_coords = [0, i] if i < 2 else [1, i - 2]
            axes[axes_coords[0]][axes_coords[1]].set_title(
                'Velicina turnira {}'.format(t_size))
            axes[axes_coords[0]][axes_coords[1]].boxplot([float_solutions[t_size]], labels=[
                'Prikaz s pomicnom tockom'])
        plt.savefig(plot_name)
        if show_plot:
            plt.show()


def find_minima_demo(f7, verbose=True):
    binary_representation = False
    limits = [-50, 150]
    genetic = GeneticAlgorithm(
        population_generation=generate_population(population_size=100, func=f7, n_vars=2,
                                                  binary_representation=binary_representation,
                                                  limits=limits, precision=3),
        num_func_calls=1e5,
        func=f7,
        selection=roulette_selection(
            elitism=True, no_elites=25),
        combination=heuristic_cross_float(),
        mutation=mutation(
            0.05, binary_representation=binary_representation, limits=limits),
        solution=solution(),
        verbose=verbose)

    best_float = genetic.evolution()
    print('Najbolji: {}'.format(best_float))

    input('Press any key to continue')


def run_all_experiments(f1, f3, f6, f7):
    zadatak_1(f1, f3, f6, f7, no_experiments=10, verbose=False,
              plot_name="prvi.png", show_plot=False)
    zadatak_2(f6, f7, no_experiments=10, verbose=False,
              plot_name="drugi.png", show_plot=False)
    zadatak_3(f6, f7, no_experiments=10, verbose=False,
              plot_name="treci.png", show_plot=False)
    zadatak_4(f6, no_experiments=10, verbose=False,
              plot_name="cetvrti.png", show_plot=False)
    zadatak_5(f6, no_experiments=10, verbose=False,
              plot_name="peti.png", show_plot=False)


def main():
    f1 = TargetFunction(lambda x1, x2: 100 * (x2 - x1 ** 2)
                        ** 2 + (1 - x1) ** 2)
    f3 = TargetFunction(
        lambda *args: sum([(args[j] - j)**2 for j in range(len(args))]))
    f6 = TargetFunction(lambda *args: .5 + (np.sin(np.sqrt(sum(np.square(args))))
                                            ** 2 - .5) / (1 + 0.001 * sum(np.square(args))) ** 2)
    f7 = TargetFunction(lambda *args: sum(np.square(args)) **
                        0.25 * (1 + np.sin(50 * (sum(np.square(args))) ** 0.1) ** 2))

    find_minima_demo(f6, verbose=True)
    zadatak_1(f1, f3, f6, f7, no_experiments=2, verbose=False,
              plot_name="prvi_demo.png", population_size=50, num_func_calls=1e4)
    zadatak_2(f6, f7, no_experiments=2, verbose=False,
              plot_name="drugi_demo.png", population_size=50, num_func_calls=1e4)
    zadatak_3(f6, f7, no_experiments=2, verbose=False,
              plot_name="treci_demo.png", population_size=50, num_func_calls=1e4)
    zadatak_4(f6, no_experiments=2, verbose=False,
              plot_name="cetvrti_demo.png", num_func_calls=1e4)
    zadatak_5(f6, no_experiments=2, verbose=False,
              plot_name="peti_demo.png", population_size=50, num_func_calls=1e4)

    #run_all_experiments(f1, f3, f6, f7)


if __name__ == "__main__":
    main()
