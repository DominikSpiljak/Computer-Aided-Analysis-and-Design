import math
from matrix import Matrix
from tabulate import tabulate
import random


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


def gradient_descent(f, f_partial, starting_point, e=1e-6, use_golden_cut=False, get_calc=False, limit_iter=-1):

    current_point = starting_point
    div_num = 0
    values = []

    gradient_calc = 0
    it = 0

    while True:

        gradient = []
        gradient_calc += 1
        for partial in f_partial:
            gradient.append(partial(*current_point))

        norm = math.sqrt(sum([grad ** 2 for grad in gradient]))

        if norm < e:
            if get_calc:
                return current_point, gradient_calc
            else:
                return current_point

        if use_golden_cut:
            gradient = [grad / norm for grad in gradient]
            def wrapper(x): return [p + x * g for p,
                                    g in zip(current_point, gradient)]
            wrap = f.wrap(wrapper)

            lamb_interval = golden_cut(0, wrap, h=1, verbose=False)
            lamb = sum(lamb_interval) / len(lamb_interval)

            current_point = [p + lamb * g for p,
                             g in zip(current_point, gradient)]

        else:
            current_point = [p + g for p, g in zip(current_point, gradient)]

        value_f = f(*current_point)

        if len(values) != 0 and value_f >= values[-1]:
            div_num += 1

        if len(values) != 0 and value_f < values[-1]:
            div_num = 0

        values.append(value_f)

        if div_num > 100:
            print(
                '\033[1m\033[91mDivergention detected, breaking loop.\033[0m\033[0m')
            return

        if limit_iter > 0 and it >= limit_iter:
            print(
                '\033[93mMax iteration {} reached for gradient descent\033[0m'.format(limit_iter))
            if get_calc:
                return current_point, gradient_calc
            else:
                return current_point

        it += 1


def newton_raphson(f, f_partial, hessian_matrix, starting_point, e=1e-6, use_golden_cut=False, get_calc=False, limit_iter=-1):

    current_point = starting_point
    div_num = 0
    values = []

    gradient_calc = 0
    hesse_calc = 0
    it = 0

    while True:

        gradient_mat = []
        hessian_mat = []

        gradient_calc += 1
        hesse_calc += 1

        for partial in f_partial:
            gradient_mat.append([partial(*current_point)])

        for row in hessian_matrix:
            hessian_row = []
            for double_partial in row:
                hessian_row.append(double_partial(*current_point))
            hessian_mat.append(hessian_row)

        gradient = Matrix()
        gradient.matrix = gradient_mat
        hessian = Matrix()
        hessian.matrix = hessian_mat

        delta_x = -1 * hessian.LUP_invert() * gradient

        delta_x = delta_x.flatten()

        norm = math.sqrt(sum([x ** 2 for x in delta_x]))

        if norm < e:
            if get_calc:
                return current_point, gradient_calc, hesse_calc
            else:
                return current_point

        if use_golden_cut:
            delta_x = [x / norm for x in delta_x]
            def wrapper(x): return [p + x * delta for p,
                                    delta in zip(current_point, delta_x)]
            wrap = f.wrap(wrapper)

            lamb_interval = golden_cut(0, wrap, h=1, verbose=False)
            lamb = sum(lamb_interval) / len(lamb_interval)

            current_point = [p + lamb * delta for p,
                             delta in zip(current_point, delta_x)]

        else:
            current_point = [p + delta for p,
                             delta in zip(current_point, delta_x)]

        value_f = f(*current_point)

        if len(values) != 0 and value_f >= values[-1]:
            div_num += 1

        if len(values) != 0 and value_f < values[-1]:
            div_num = 0

        values.append(value_f)

        if div_num > 100:
            print(
                '\033[1m\033[91mDivergention detected, breaking loop.\033[0m\033[0m')
            return

        if limit_iter > 0 and it >= limit_iter:
            print(
                '\033[93mMax iteration {} reached for newton raphson\033[0m'.format(limit_iter))
            if get_calc:
                return current_point, gradient_calc, hesse_calc
            else:
                return current_point

        it += 1


def no_restriction_transformation(starting_point, f, gs=[], hs=[], dx=.5, e=1e-6, t=1):
    values = []
    div_num = 0

    def inner_point_func(f, gs):
        def func(*x):
            return -sum([g(*x) * int(g(*x) < 0) for g in gs])
        return func

    if any([g(*starting_point) < 0 for g in gs]):
        X0 = hooke_jeeves(
            starting_point, inner_point_func(f, gs), verbose=False)
    else:
        X0 = starting_point

    while True:
        def func(*x):
            try:
                val = f(*x) - (1 / t) * sum([math.log(g(*x))
                                             for g in gs]) + t * sum([h(*x) ** 2 for h in hs])
                return val
            except ValueError:
                return math.inf

        X0_n = hooke_jeeves(X0, func, verbose=False)
        value_n = f(*X0_n)
        if math.sqrt(sum([(x0_ - x0_n_)**2 for x0_, x0_n_ in zip(X0, X0_n)])) < e:
            return X0_n

        t *= 10

        if len(values) != 0 and value_n >= values[-1]:
            div_num += 1

        if len(values) != 0 and value_n < values[-1]:
            div_num = 0

        values.append(value_n)
        X0 = X0_n

        if div_num > 100:
            print(
                '\033[1m\033[91mDivergention detected, breaking loop.\033[0m\033[0m')
            return


def box_optimizer(f, starting_point, gs=[], x_range=[-1000, 1000], alpha=1.3, epsilon=1e-6, limit_iter=-1):
    X0 = starting_point

    for x in X0:
        if x > x_range[1] or x < x_range[0]:
            raise ValueError(
                "Starting point doesn't meet x_range requirements")

    if len(gs) != 0:
        for g in gs:
            if g(*X0) < 0:
                raise ValueError("Starting point doesn't meet gs requirements")

    n = len(X0)

    Xc = X0
    Xes = [X0]
    for _ in range(2 * len(Xc) - 1):
        r = random.random()
        X_new = [x_range[0] + r * (x_range[1] - x_range[0])] * len(Xc)

        while any([g(*X_new) < 0 for g in gs]):
            X_new = [(1 / 2) * (x + c) for x, c in zip(X_new, Xc)]

        Xes.append(X_new)

        Xc = [sum([x[i] for x in Xes]) / len(Xes) for i in range(n)]

    it = 0
    while True:
        f_vals = [f(*x) for x in Xes]
        h = f_vals.index(sorted(f_vals, reverse=True)[0])
        h2 = f_vals.index(sorted(f_vals, reverse=True)[1])

        Xc = [sum([x[i] for x in Xes if Xes.index(x) != h]) /
              (len(Xes) - 1) for i in range(n)]

        Xr = [(1 + alpha) * Xc[i] - alpha * Xes[h][i] for i in range(n)]

        for i in range(n):
            if Xr[i] < x_range[0]:
                Xr[i] = x_range[0]
            elif Xr[i] > x_range[1]:
                Xr[i] = x_range[1]

        while any([g(*Xr) < 0 for g in gs]):
            Xr = [(1 / 2) * (x + c) for x, c in zip(Xr, Xc)]

        if f(*Xr) > f(*Xes[h2]):
            Xr = [(1 / 2) * (x + c) for x, c in zip(Xr, Xc)]

        Xes[h] = Xr

        if math.sqrt((1/len(Xes)) * sum([(f(*Xes[i]) - f(*Xc)) ** 2 for i in range(len(Xes))])) <= epsilon:
            f_vals = [f(*x) for x in Xes]
            return Xes[f_vals.index(min(f_vals))]

        if limit_iter > 0 and it >= limit_iter:
            print(
                '\033[93mMax iteration {} reached for box optimizer\033[0m'.format(limit_iter))
            f_vals = [f(*x) for x in Xes]
            return Xes[f_vals.index(min(f_vals))]

        it += 1


def print_task_num(num):
    print()
    print('#' * 40, 'zadatak ', num, '#' * 40)
    print()


def main():
    random.seed(40)
    f1 = TargetFunction(lambda x1, x2: 100 * (x2 - x1 ** 2)
                        ** 2 + (1 - x1) ** 2)
    f1_partial = [lambda x1, x2: 2 * (200 * x1 ** 3 - 200 * x1 * x2 + x1 - 1),
                  lambda x1, x2: 200 * (x2 - x1 ** 2)]
    f1_hessian = [[lambda x1, x2: 2 * (600 * x1 ** 2 - 200 * x2 + 1), lambda x1, x2: -400 * x1],
                  [lambda x1, x2: -400 * x1, lambda x1, x2: 200]]
    f2 = TargetFunction(lambda x1, x2: (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2)
    f2_partial = [lambda x1, x2: 2 * (x1 - 4), lambda x1, x2: 8 * (x2 - 2)]
    f2_hessian = [[lambda x1, x2: 2, lambda x1, x2: 0],
                  [lambda x1, x2: 0, lambda x1, x2: 8]]
    f3 = TargetFunction(lambda x1, x2: (x1 - 2) ** 2 + (x2 + 3) ** 2)
    f3_partial = [lambda x1, x2: 2 * (x1 - 2), lambda x1, x2: 2 * (x2 + 3)]
    f3_hessian = [[lambda x1, x2: 2, lambda x1, x2: 0],
                  [lambda x1, x2: 0, lambda x1, x2: 2]]
    f4 = TargetFunction(lambda x1, x2: (x1 - 3) ** 2 + x2 ** 2)

    ######################################## zadatak  1 ########################################
    print_task_num(1)
    print('Gradijentni spust bez određivanja optimalnog iznosa')
    gradient_descent(f3, f3_partial, [0, 0], use_golden_cut=False)
    f3.reset()
    print('Gradijentni spust sa određivanjem optimalnog iznosa')
    min_x, gradient_calc = gradient_descent(
        f3, f3_partial, [0, 0], use_golden_cut=True, get_calc=True)
    print('Nađen minimum za točku {} i vrijednost funkcije {}. Broj računanja gradijenta: {}'.format(
        min_x, f3(*min_x), gradient_calc))

    ######################################## zadatak  2 ########################################
    print_task_num(2)
    print('Function 1')
    gradient_min, gradient_calc = gradient_descent(
        f1, f1_partial, [-1.9, 2], use_golden_cut=True, get_calc=True, limit_iter=10000)
    gradient = [gradient_min, f1(*gradient_min), gradient_calc, 0, f1.no_calls]
    f1.reset()
    newton_min, newton_grad_calc, newton_hesse_calc = newton_raphson(
        f1, f1_partial, f1_hessian, [-1.9, 2], use_golden_cut=True, get_calc=True, limit_iter=10000)
    newton = [newton_min, f1(*newton_min), newton_grad_calc,
              newton_hesse_calc, f1.no_calls]
    f1.reset()

    print(tabulate([['Gradient descent'] + gradient, ['Newton-Raphson'] + newton],
                   headers=['', 'Minimum point', 'Minimum value', 'Number of gradient calculations', 'Number of Hessian calculations', 'Function calls'], tablefmt='orgtbl'))

    print()
    print('Function 2')
    gradient_min, gradient_calc = gradient_descent(
        f2, f2_partial, [0.1, 0.3], use_golden_cut=True, get_calc=True)
    gradient = [gradient_min, f2(*gradient_min), gradient_calc, 0, f2.no_calls]
    f2.reset()
    newton_min, newton_grad_calc, newton_hesse_calc = newton_raphson(
        f2, f2_partial, f2_hessian, [0.1, 0.3], use_golden_cut=True, get_calc=True)
    newton = [newton_min, f2(*newton_min), newton_grad_calc,
              newton_hesse_calc, f2.no_calls]
    f2.reset()

    print(tabulate([['Gradient descent'] + gradient, ['Newton-Raphson'] + newton],
                   headers=['', 'Minimum point', 'Minimum value', 'Number of gradient calculations', 'Number of Hessian calculations', 'Function calls'], tablefmt='orgtbl'))

    ######################################## zadatak  3 ########################################
    print_task_num(3)

    gs = [lambda x1, x2: x2 - x1, lambda x1, x2: 2 - x1]
    x_range = [-100, 100]

    print('Function 1')
    min_x = box_optimizer(f1, [-1.9, 2], gs, x_range)
    print('Nađen minimum za točku {} i vrijednost funkcije {}.'.format(
        min_x, f1(*min_x)))
    f1.reset()

    print('Function 2')
    min_x = box_optimizer(f2, [0.1, 0.3], gs, x_range)
    print('Nađen minimum za točku {} i vrijednost funkcije {}.'.format(
        min_x, f2(*min_x)))
    f2.reset()

    ######################################## zadatak  4 ########################################
    print_task_num(4)
    gs = [lambda x1, x2: x2 - x1, lambda x1, x2: 2 - x1]

    print('Function 1')
    min_x = no_restriction_transformation([-1.9, 2], f1, gs=gs)
    print('Nađen minimum s početnom točkom {} za točku {} i vrijednost funkcije {}.'.format(
        [-1.9, 2], min_x, f1(*min_x)))

    print('Traženje iz bolje početne točke...')

    min_x = no_restriction_transformation([3, 3], f1, gs=gs)
    print('Nađen minimum s početnom točkom {} za točku {} i vrijednost funkcije {}.'.format(
        [3, 3], min_x, f1(*min_x)))

    print()
    print('Function 2')
    min_x = no_restriction_transformation([0.1, 0.3], f2, gs=gs)
    print('Nađen minimum s početnom točkom {} za točku {} i vrijednost funkcije {}.'.format(
        [0.1, 0.3], min_x, f2(*min_x)))

    print('Nađen je najbolji mogući minimum uz restrikcije')

    ######################################## zadatak  5 ########################################
    print_task_num(5)
    gs = [lambda x1, x2: 3 - x1 - x2, lambda x1,
          x2: 3 + 1.5 * x1 - x2, lambda x1, x2: x2 - 1]

    min_x = no_restriction_transformation([5, 5], f3, gs=gs)
    print('Nađen minimum s početnom točkom {} za točku {} i vrijednost funkcije {}.'.format(
        [5, 5], min_x, f3(*min_x)))


if __name__ == "__main__":
    main()
