import numpy as np
import abc
from matrix import Matrix


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

        derivation_value = linear_system.calculate(
            linear_system.initial_state, 0)

        t = self.integration_period
        i = 0

        while t <= self.max_time:

            predicted_value = self.calculate_step(
                predicted_values, derivation_value)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('System state t={}: \nx={}'.format(
                        t, predicted_value))

            derivation_value = linear_system.calculate(predicted_value, t)

            t = round(t + self.integration_period, 8)
            i += 1

        print('System state t={}: \nx={}'.format(
            self.max_time, predicted_value))
        return np.array(predicted_values)

    def calculate_step(self, predicted_values, derivation_value):
        delta_x = self.integration_period * derivation_value
        return predicted_values[-1] + delta_x


class InverseEulerPredictor(Predictor):

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        P = np.linalg.inv(
            1 - linear_system.system_matrix * self.integration_period)
        Q = P * self.integration_period * linear_system.external_stimulus_matrix

        print(linear_system.system_matrix * self.integration_period)
        print(P)

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

        print('System state t={}: \nx={}'.format(
            self.max_time, predicted_value))

        return np.array(predicted_values)

    def calculate_step(self, predicted_values, P, Q, external_stimulus, t):
        delta_x = P.dot(predicted_values[-1]) + Q.dot(external_stimulus(t))
        return predicted_values[-1] + delta_x


def main():
    ls = LinearSystem(np.array([[0, 1], [-200, -102]]), np.array([[1], [-2]]))
    ep = EulerPredictor(0.01, 0.02)

    ep.predict(ls)

    ep = InverseEulerPredictor(0.01, 0.02)

    print(ep.predict(ls))


if __name__ == "__main__":
    main()
