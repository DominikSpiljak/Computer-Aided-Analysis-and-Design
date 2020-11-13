import math
from tabulate import tabulate
import random
from tqdm import tqdm


class TargetFunction:
    """Function class used for optimization, counts calls and saves calculated results
    """

    def __init__(self, f):
        self.f = f
        self.results = {}
        self.no_calls = 0
        self.params = None
        self.missing = None

    def __call__(self, *args):
        """Method that overrides and handles calling object as function

        Returns:
            int: function value for given parameters
        """
        self.no_calls += 1

        if self.params:
            self.params[self.missing] = args[0]
            args = tuple([self.params[key] for key in sorted(self.params)])

        if args in self.results:
            return self.results[args]
        else:
            result = self.f(*args)
            self.results[args] = result
            return result

    def reset_counter(self):
        self.no_calls = 0

    def reset(self):
        self.reset_counter()
        self.params = None
        self.missing = None

    def predefine_params(self, params, missing):
        self.params = params
        self.missing = missing


def unimodal_interval(point, h, f):
    """Finds interval that contains 1-D function minimum

    Args:
        point ([float]): [starting point]
        h ([float]): [step size]
        f ([TargetFunction]): [function for which interval needs to be found]

    Returns:
        [list]: [interval containing minimum]
    """
    l = point - h
    r = point + h
    m = point
    step = 1

    fm = f(point)
    fl = f(l)
    fr = f(r)

    if fm < fr and fm < fl:
        pass

    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = f(r)

    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            fl = f(l)

    return [l, r]


def golden_cut(start, f, e=1e-6, h=None, verbose=True):
    """[Performs golden cut to narrow the interval]

    Args:
        start ([list or float]): [list if interval is already found or float to perform unimodal interval func (requires h to be defined)]
        f ([TargetFunction]): [function for which interval needs to be found]
        e ([list], optional): [maximal size of interval]. Defaults to 10e-6.
        h ([float], optional): [step size]. Defaults to None.

    Raises:
        ValueError: [in case float is given as start variable and h is not defined]

    Returns:
        [interval]: [interval containing minimum]
    """
    if type(start) != list:
        if h is None:
            raise ValueError(
                'No interval was given but step h not defined for unimodal interval')

        interval = unimodal_interval(start, h, f)

    else:
        interval = start

    k = .5 * (math.sqrt(5) - 1)
    a = interval[0]
    b = interval[1]
    c = b - k * (b - a)
    d = a + k * (b - a)

    fc = f(c)
    fd = f(d)

    iteration = 0

    if verbose:
        print('Iteration {}:'.format(iteration))
        print('a = {}, f(a) = {} | c = {}, f(c) = {} | d = {}, f(d) = {} | b = {}, f(b) = {}'.format(
            a, f(a), c, f(c), d, f(d), b, f(b)))

    while (b - a) > e:

        iteration += 1

        if fc < fd:
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = f(c)

        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = f(d)

        if verbose:
            print('Iteration {}:'.format(iteration))
            print('a = {}, f(a) = {} | c = {}, f(c) = {} | d = {}, f(d) = {} | b = {}, f(b) = {}'.format(
                a, f(a), c, f(c), d, f(d), b, f(b)))

    return [a, b]


def hooke_jeeves(point, f, dx=.5, e=1e-6, verbose=True):
    """[Performs Hooke Jeeves method to find minimum of the function]

    Args:
        point ([list]): [starting point]
        f ([TargetFunction]): [function for which minimum needs to be found]
        dx (float, optional): [step size]. Defaults to .5.
        e ([type], optional): [stopping criteria]. Defaults to 10e-6.
    Returns:
        [list]: [minimum point find using this method]
    """
    def find(xP, f, dx):
        xN = xP.copy()
        for i in range(len(xP)):
            p = f(*xN)
            xN[i] += dx
            n = f(*xN)

            if n > p:
                xN[i] -= 2 * dx
                n = f(*xN)

                if n > p:
                    xN[i] += dx

        return xN

    xB = point.copy()
    xP = point.copy()

    iteration = 0

    while dx > e:
        iteration += 1

        xN = find(xP, f, dx)

        if verbose:
            print('Iteration {}:'.format(iteration))
            print('\txB = {}, f(xB) = {} | xP = {}, f(xP) = {} | xN = {}, f(xN) = {} || f(xB) > f(xN): {} {}'.format(xB, f(*xB),
                                                                                                                     xP, f(
                                                                                                                         *xP),
                                                                                                                     xN, f(
                                                                                                                         *xN),
                                                                                                                     f(*xB) > f(
                                                                                                                         *xN),
                                                                                                                     ', dx = {}'.format(dx / 2) if f(*xB) <= f(*xN) else ''))

        if f(*xN) < f(*xB):
            xP = [2 * xN[i] - xB[i] for i in range(len(xN))]
            xB = xN

        else:
            dx /= 2
            xP = xB

    return xB


def nelder_mead_simplex(point, f, dx=1, e=1e-6, alpha=1, beta=0.5, gamma=2, sigma=.5, verbose=True):
    simplex_points = [point.copy()]
    for i in range(len(point)):
        simplex = point.copy()
        simplex[i] += dx
        simplex_points.append(simplex)

    iteration = 0

    while True:
        # Calculate h and l
        simplex_values = [f(*simplex) for simplex in simplex_points]
        h = simplex_values.index(max(simplex_values))
        l = simplex_values.index(min(simplex_values))

        # Calculate centroid
        no_h_simplex_points = [simplex_points[i]
                               for i in range(len(simplex_points)) if i != h]
        Xc = []
        for i in range(len(point)):
            dim_point = 0
            for point in no_h_simplex_points:
                dim_point += point[i]
            Xc.append(dim_point / len(no_h_simplex_points))

        if verbose:
            print('Iteration: {}'.format(iteration))
            print('Centroid = {}, f(Centroid) = {}'.format(Xc, f(*Xc)))
        iteration += 1

        # Reflect
        Xr = [(1 + alpha) * Xc[i] - alpha * simplex_points[h][i]
              for i in range(len(Xc))]

        if f(*Xr) < f(*simplex_points[l]):
            # Expand
            Xe = [(1 - gamma) * Xc[i] + gamma * Xr[i] for i in range(len(Xc))]

            if f(*Xe) < f(*simplex_points[l]):
                simplex_points[h] = Xe
            else:
                simplex_points[h] = Xr

        else:
            if all([f(*Xr) > f(*point) for point in no_h_simplex_points]):
                if f(*Xr) < f(*simplex_points[h]):
                    simplex_points[h] = Xr

                # Contract
                Xk = [(1 - beta) * Xc[i] + beta * simplex_points[h][i]
                      for i in range(len(Xc))]

                if f(*Xk) < f(*simplex_points[h]):
                    simplex_points[h] = Xk

                else:
                    for i in range(len(simplex_points)):
                        simplex_points[i] = [
                            (1 / 2) * (simplex_points[i][j] + simplex_points[l][j]) for j in range(len(simplex_points[i]))]
            else:
                simplex_points[h] = Xr

        if math.sqrt((1/len(simplex_points)) * sum([(f(*simplex_points[i]) - f(*Xc)) ** 2 for i in range(len(simplex_points))])) <= e:
            Xc = []
            for k in range(len(point)):
                dim_point = 0
                for simplex_point in simplex_points:
                    dim_point += simplex_point[k]
                Xc.append(dim_point / len(simplex_points))
            return Xc


def coordinate_axes_search(point, f, e=1e-6, h=1, verbose=True):
    x = point.copy()
    while True:
        xs = x.copy()
        for i in range(len(x)):
            f.predefine_params({j: x[j] for j in range(len(x)) if j != i}, i)
            interval = golden_cut(x[i], f, h=h, e=e, verbose=verbose)
            lam = sum(interval) / len(interval)
            x[i] = lam
        if math.sqrt(sum([(x[i] - xs[i])**2 for i in range(len(x))])) <= e:
            return x


def print_task_num(num):
    print()
    print('#' * 40, 'zadatak ', num, '#' * 40)
    print()


def main():
    def f1(x, y): return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
    def f2(x, y): return (x - 4) ** 2 + 4 * (y - 2) ** 2
    f3 = lambda *args: sum([(args[j] - j)**2 for j in range(len(args))])
    def f4(x, y): return abs((x - y) * (x + y)) + math.sqrt(x ** 2 + y ** 2)
    f6 = lambda *args: .5 + ((math.sin(sum([x ** 2 for x in args])) ** 2 - .5) / (
        1 + .001 * sum([x ** 2 for x in args])) ** 2)

    ######################################## zadatak  1 ########################################
    print_task_num(1)
    tf = TargetFunction(lambda x: (x - 3) ** 2)
    for point in [10, 50, 100]:
        print('####### Početna točka: {} #######'.format(point))
        print('Zlatni rez: ')
        print('Interval: {}, broj evaluacija funkcije: {}'.format(
            golden_cut(point, tf, h=1, verbose=False), tf.no_calls))
        print()
        tf.reset_counter()

        print('Pretraživanje po koordinatnim osima: ')
        print('Točka: {}, broj evaluacija funkcije: {}'.format(
            coordinate_axes_search([point], tf, h=1, verbose=False), tf.no_calls))
        print()
        tf.reset_counter()

        print('Simpleks postupak po Nelderu i Meadu: ')
        print('Centroid: {}, broj evaluacija funkcije: {}'.format(
            nelder_mead_simplex([point], tf, verbose=False), tf.no_calls))
        print()
        tf.reset_counter()

        print('Postupak Hooke-Jeeves: ')
        print('Točka: {}, broj evaluacija funkcije: {}'.format(
            hooke_jeeves([point], tf, verbose=False), tf.no_calls))
        print()
        tf.reset_counter()

    ######################################## zadatak  2 ########################################
    print_task_num(2)
    tf1 = TargetFunction(f1)
    tf2 = TargetFunction(f2)
    tf3 = TargetFunction(f3)
    tf4 = TargetFunction(f4)
    funcs = [tf1, tf2, tf3, tf4]
    starting_points = [[-1.9, 2], [.1, .3], [0, 0, 0, 0, 0], [5.1, 1.1]]
    optimizers = [nelder_mead_simplex, hooke_jeeves, coordinate_axes_search]
    minimums = [[], [], []]
    func_calls = [[], [], []]
    for f, point in zip(funcs, starting_points):
        for i, optimizer in enumerate(optimizers):
            min_point = optimizer(point, f, verbose=False)
            func_calls[i].append(f.no_calls)
            f.reset()
            minimums[i].append(f(*min_point))

    print('Minimumi: ')
    print(tabulate([['Nelder-Mead simplex'] + minimums[0], ['Hooke-Jeeves'] + minimums[1], ['Pretraživanje po koordinatnim osima'] + minimums[2]],
                   headers=['Optimizator \ Function', 'f1', 'f2', 'f3', 'f4'], tablefmt='orgtbl'))
    print()
    print('Broj evaluacija funkcije: ')
    print(tabulate([['Nelder-Mead simplex'] + func_calls[0], ['Hooke-Jeeves'] + func_calls[1], ['Pretraživanje po koordinatnim osima'] + func_calls[2]],
                   headers=['Optimizator \ Function', 'f1', 'f2', 'f3', 'f4'], tablefmt='orgtbl'))

    ######################################## zadatak  3 ########################################
    print_task_num(3)
    optimizers = [nelder_mead_simplex, hooke_jeeves, coordinate_axes_search]
    optimizer_names = ['Nelder-Mead simplex',
                       'Hooke-Jeeves', 'Pretraživanje po koordinatnim osima']
    tf4.reset()
    for optimizer, optimizer_name in zip(optimizers, optimizer_names):
        print('Pokrecem {}'.format(optimizer_name))
        min_point = optimizer([5, 5], tf4, verbose=False)
        tf4.reset()
        print('Nađen minimum za točku {} sa vrijednosti {}'.format(
            min_point, tf4(*min_point)))
        print()

    ######################################## zadatak  4 ########################################
    print_task_num(4)
    tf1.reset()
    points = [[.5, .5], [20, 20]]
    minimums = [[], []]
    func_calls = [[], []]
    dxs = [dx for dx in range(1, 20, 5)]

    for i, point in enumerate(points):
        for dx in dxs:
            min_point = nelder_mead_simplex(point, tf1, dx=dx, verbose=False)
            func_calls[i].append(tf1.no_calls)
            tf1.reset()
            minimums[i].append(tf1(*min_point))

    print('Minimumi: ')
    print(tabulate([['[0.5, 0.5]'] + minimums[0], ['[20, 20]'] + minimums[1]],
                   headers=['Point \ dx'] + dxs, tablefmt='orgtbl'))
    print()
    print('Broj evaluacija funkcije: ')
    print(tabulate([['[0.5, 0.5]'] + func_calls[0], ['[20, 20]'] + func_calls[1]],
                   headers=['Point \ dx'] + dxs, tablefmt='orgtbl'))

    ######################################## zadatak  5 ########################################
    print_task_num(5)
    tf6 = TargetFunction(f6)
    fmin = 0
    niters = 20000
    points = []
    for _ in range(niters):
        points.append([random.random() * 100 - 50, random.random() * 100 - 50])

    found = 0

    for point in tqdm(points):
        min_point = nelder_mead_simplex(point, tf6, verbose=False)
        if abs(tf6(*min_point)) < 10e-4:
            found += 1

    print('Vjerojatnost da je globalni optimum nađen koristeći Nelder-Mead simplex koristeći {} nasumičnih točaka je {}'.format(niters, found / niters))


if __name__ == "__main__":
    main()
