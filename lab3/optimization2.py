import math
from matrix import Matrix


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


def gradient_descent(f, f_partial, starting_point, e=1e-6, use_golden_cut=False):

    current_point = starting_point
    div_num = 0
    values = []

    while True:

        gradient = []

        for partial in f_partial:
            gradient.append(partial(*current_point))

        norm = math.sqrt(sum([grad ** 2 for grad in gradient]))

        if norm < e:
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

        values.append(value_f)

        if div_num > 100:
            print('Divergention detected, breaking loop.')
            break


def newton_raphson(f, f_partial, hessian_matrix, starting_point, e=1e-6, use_golden_cut=False):

    current_point = starting_point
    div_num = 0
    values = []

    while True:

        gradient_mat = []
        hessian_mat = []

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

        values.append(value_f)

        if div_num > 100:
            print('Divergention detected, breaking loop.')
            break


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
                val = f(*x) - (1 / t) * \
                    sum([math.log(g(*x)) for g in gs]) + \
                    t * sum([h(*x) ** 2 for h in hs])
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

        values.append(value_n)
        X0 = X0_n

        if div_num > 100:
            pass
            print('Divergention detected, breaking loop.')
            break


def box_optimizer():
    pass


def main():
    f = TargetFunction(lambda x1, x2: (x1 - 4) ** 2 + 4 * (x2 - 2) ** 2)
    min_x = gradient_descent(f, [lambda x1, x2: 2 * (x1 - 4), lambda x1,
                                 x2: 8 * (x2 - 2)], starting_point=[0, 0], use_golden_cut=True)
    print(min_x)
    print(f.no_calls)
    f.reset()

    min_x = newton_raphson(f, [lambda x1, x2: 2 * (x1 - 4), lambda x1, x2: 8 * (x2 - 2)],
                           [[lambda x1, x2: 2, lambda x1, x2: 0],
                            [lambda x1, x2: 0, lambda x1, x2: 8]],
                           starting_point=[0, 0], use_golden_cut=True)
    print(min_x)
    print(f.no_calls)

    min_x = no_restriction_transformation(
        [2, 1], f, hs=[lambda x1, x2: x2 - x1])

    print(min_x)
    print(f.no_calls)


if __name__ == "__main__":
    main()
