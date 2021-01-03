import numpy as np
import abc


class LinearSystem:

    def __init__(self, system_matrix, initial_state, external_stimulus_matrix=None, external_stimulus=None):
        self.system_matrix = system_matrix
        self.initial_state = initial_state
        if external_stimulus_matrix is None or external_stimulus is None:
            self.external_stimulus_matrix = np.zeros(self.system_matrix.shape)
            self.external_stimulus = lambda x: np.zeros(
                self.initial_state.shape)

    def calculate(self, x, t):
        return self.system_matrix.dot(x) + self.external_stimulus_matrix.dot(self.external_stimulus(t))


class Predictor:
    def __init__(self, integration_period, max_time):
        self.integration_period = integration_period
        self.max_time = max_time

    @abc.abstractmethod
    def predict(self, linear_system):
        pass

    @abc.abstractmethod
    def calculate_step(self, predicted_values, derivation_value):
        pass


class EulerPredictor(Predictor):

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        t = self.integration_period
        i = 0

        while t <= self.max_time:

            predicted_value = self.calculate_step(
                predicted_values, t, linear_system)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('Euler prediction:\nSystem state t={}: \nx={}'.format(
                        t, predicted_value))

            t = round(t + self.integration_period, 8)
            i += 1

        print('Euler prediction:\nSystem state t={}: \nx={}'.format(
            self.max_time, predicted_value))
        return np.array(predicted_values)

    def calculate_step(self, predicted_values, t, linear_system):
        derivation_value = linear_system.calculate(predicted_values[-1], t - self.integration_period)

        delta_x = self.integration_period * derivation_value
        return predicted_values[-1] + delta_x


class InverseEulerPredictor(Predictor):

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        P = np.linalg.inv(np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix *
                          self.integration_period)
        Q = P * self.integration_period * linear_system.external_stimulus_matrix

        t = self.integration_period
        i = 0

        while t <= self.max_time:

            predicted_value = self.calculate_step(
                predicted_values, P, Q, linear_system.external_stimulus, t)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('System state t={}: \nx={}'.format(
                        t, predicted_value))

            t = round(t + self.integration_period, 8)
            i += 1

        print('Inverse Euler prediction:\nSystem state t={}: \nx={}'.format(
            self.max_time, predicted_value))

        return np.array(predicted_values)

    def calculate_step(self, predicted_values, P, Q, external_stimulus, t):
        return P.dot(predicted_values[-1]) + Q.dot(external_stimulus(t))


class TrapezoidalPredictor(Predictor):

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        R = (np.linalg.inv(
            (np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix * 1/2 * self.integration_period)).dot((
                np.eye(*linear_system.system_matrix.shape) + linear_system.system_matrix * 1/2 * self.integration_period)))

        S = (np.linalg.inv(
            (np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix * 1/2 * self.integration_period))
            * self.integration_period/2 * linear_system.external_stimulus_matrix)

        t = self.integration_period
        i = 0

        while t <= self.max_time:

            predicted_value = self.calculate_step(
                predicted_values, R, S, linear_system.external_stimulus, t)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('System state t={}: \nx={}'.format(
                        t, predicted_value))

            t = round(t + self.integration_period, 8)
            i += 1

        print('Trapezoidal prediction:\nSystem state t={}: \nx={}'.format(
            self.max_time, predicted_value))

        return np.array(predicted_values)

    def calculate_step(self, predicted_values, R, S, external_stimulus, t):
        return R.dot(predicted_values[-1]) + S.dot(external_stimulus(t - self.integration_period) + external_stimulus(t))


class RungeKutta4Predictor(Predictor):

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        t = self.integration_period
        i = 0

        while t <= self.max_time:

            predicted_value = self.calculate_step( predicted_values, t, linear_system)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('RungeKutta prediction:\nSystem state t={}: \nx={}'.format(
                        t, predicted_value))

            t = round(t + self.integration_period, 8)
            i += 1

        print('RungeKutta prediction:\nSystem state t={}: \nx={}'.format(
            self.max_time, predicted_value))
        return np.array(predicted_values)

    def calculate_step(self, predicted_values, t, linear_system):
        tk = t - self.integration_period
        m1 = linear_system.calculate(predicted_values[-1], tk)
        m2 = linear_system.calculate(predicted_values[-1] + self.integration_period / 2 * m1, tk + self.integration_period / 2)
        m3 = linear_system.calculate(predicted_values[-1] + self.integration_period / 2 * m2, tk + self.integration_period / 2)
        m4 = linear_system.calculate(predicted_values[-1] + self.integration_period * m3, tk + self.integration_period)
        return predicted_values[-1] + self.integration_period / 6 * (m1 + 2 * m2 + 2 * m3 + m4)

def main():
    ls = LinearSystem(np.array([[0, 1], [-200, -102]]), np.array([[1], [-2]]))

    s_e = (0.01, 0.1)

    ep = EulerPredictor(*s_e)
    ep.predict(ls)

    iep = InverseEulerPredictor(*s_e)
    iep.predict(ls)

    tp = TrapezoidalPredictor(*s_e)
    tp.predict(ls)

    rk = RungeKutta4Predictor(*s_e)
    rk.predict(ls)

if __name__ == "__main__":
    main()
