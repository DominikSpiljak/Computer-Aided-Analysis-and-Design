import math

class GoalFunction:
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
        
def unimodal_solution(point, h, f):
    l = point - h
    r = point + h
    m = point
    step = 1

    fm = f(point);
    fl = f(l);
    fr = f(r);

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

def golden_cut(start, f, e=10e-6, h=None):
    if type(start) != list:
        if h is None:
            raise ValueError('No interval was given but step h not defined for unimodal solution')

        interval = unimodal_solution(start, h, f)
    
    else:
        interval = start

    k = .5 * (math.sqrt(5) - 1)
    a = interval[0]
    b = interval[1]
    c = b - k * (b - a)
    d = a + k * (b - a)

    fc = f(c)
    fd = f(d)

    while (b - a) > e:
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
    
    return [a, b]

def main():
    gfunc = GoalFunction(lambda x: (x - 4)**2)
    print(golden_cut(0, gfunc, h=1))

if __name__ == "__main__":
    main()
