import numpy as np
import abc
import matplotlib.pyplot as plt
import math


class LinearSystem:

    def __init__(self, system_matrix, initial_state, external_stimulus_matrix=None, external_stimulus=None):
        self.system_matrix = system_matrix
        self.initial_state = initial_state
        if external_stimulus_matrix is None or external_stimulus is None:
            self.external_stimulus_matrix = np.zeros(self.system_matrix.shape)
            self.external_stimulus = lambda x: np.zeros(
                self.initial_state.shape)
        else:
            self.external_stimulus_matrix = external_stimulus_matrix
            self.external_stimulus = external_stimulus

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

    def init_implicit_matrices(self, linear_system):
        P = np.linalg.inv(np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix *
                          self.integration_period)
        Q = P * self.integration_period * linear_system.external_stimulus_matrix

        return P, Q

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        P, Q = self.init_implicit_matrices(linear_system)

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

    def calculate_with_future_value(self, predicted_values, future_derivation, linear_system):
        return predicted_values[-1] + self.integration_period * future_derivation


class TrapezoidalPredictor(Predictor):

    def init_implicit_matrices(self, linear_system):
        R = (np.linalg.inv(
            (np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix * 1/2 * self.integration_period)).dot((
                np.eye(*linear_system.system_matrix.shape) + linear_system.system_matrix * 1/2 * self.integration_period)))

        S = (np.linalg.inv(
            (np.eye(*linear_system.system_matrix.shape) - linear_system.system_matrix * 1/2 * self.integration_period))
            * self.integration_period/2 * linear_system.external_stimulus_matrix)

        return R, S

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        R, S = self.init_implicit_matrices(linear_system)
       
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

    def calculate_with_future_value(self, predicted_values, future_derivation, linear_system):
        return predicted_values[-1] + self.integration_period / 2 * (future_derivation + linear_system.calculate(predicted_values[-1],
         self.integration_period * (len(predicted_values) - 1)))


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
                    print('System state t={}: \nx={}'.format(
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


class PredictionCorrectionPredictor(Predictor):
    
    def __init__(self, integration_period, max_time, predictor, corrector, corrector_times=1, name=""):
        super(PredictionCorrectionPredictor, self).__init__(integration_period, max_time)
        self.predictor = predictor(integration_period, max_time)
        self.corrector = corrector(integration_period, max_time)
        self.corrector_times = corrector_times
        self.name = name

    def predict(self, linear_system, num_iter_print=None):
        predicted_values = [linear_system.initial_state]

        t = self.integration_period
        i = 0
        
        while t <= self.max_time:

            predicted_value = self.calculate_step(predicted_values, t, linear_system)
            predicted_values.append(predicted_value)

            if num_iter_print is not None:
                if (i + 1) % num_iter_print == 0:
                    print('{} prediction:\nSystem state t={}: \nx={}'.format(
                        self.name, t, predicted_value))

            t = round(t + self.integration_period, 8)
            i += 1

        print('{} prediction:\nSystem state t={}: \nx={}'.format(
            self.name, self.max_time, predicted_value))
        return np.array(predicted_values)


    def calculate_step(self, predicted_values, t, linear_system):
        pred = self.predictor.calculate_step(predicted_values, t, linear_system)
        for _ in range(self.corrector_times):
            pred = self.corrector.calculate_with_future_value(predicted_values, linear_system.calculate(pred, t - self.integration_period), linear_system)
        return pred


def print_task_num(num):
    print()
    print('#' * 40, 'zadatak ', num, '#' * 40)
    print()


def init_predictors(integration_period, max_time):
    predictors = []
    predictors.append(EulerPredictor(integration_period, max_time))
    predictors.append(InverseEulerPredictor(integration_period, max_time))
    predictors.append(TrapezoidalPredictor(integration_period, max_time))
    predictors.append(RungeKutta4Predictor(integration_period, max_time))
    predictors.append(PredictionCorrectionPredictor(integration_period, max_time, predictor=EulerPredictor, corrector=InverseEulerPredictor, corrector_times=2, name="PE(CE)^2"))
    predictors.append(PredictionCorrectionPredictor(integration_period, max_time, predictor=EulerPredictor, corrector=TrapezoidalPredictor, corrector_times=1, name="PECE"))
    return predictors

def predict(predictors, linear_system):
    predictions =  []
    for predictor in predictors:
        predictions.append(predictor.predict(linear_system))
    return predictions

def plot_predictions(predictions, integration_period, max_time, task_num):
    predictor_names = ["Euler predictor", "Inverse Euler predictor", "Trapezoidal predictor",
     "Runge Kutta 4-degree predictor", "PE(CE)^2 predictor with: predictor = Euler, corrector = Inverse Euler",
      "PECE predictor with: predictor = Euler, corrector = Trapezoidal"]
    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(18, 10)
    fig.suptitle('Task {}, variable predictions, T = {}'.format(task_num, integration_period))
    time = np.linspace(0, max_time, predictions[0].shape[0])
    for i, prediction in enumerate(predictions):
        indices = [i // 2, i % 2]
        axes[indices[0], indices[1]].set_title(predictor_names[i])
        for j in range(prediction.shape[1]):
            axes[indices[0], indices[1]].plot(time, prediction[:, j], label='x[{}]'.format(j))
        axes[indices[0], indices[1]].legend()
    fig.tight_layout()
    plt.show()

def plot_errs(true, predictions, integration_period, max_time, task_num):
    errs = []
    for prediction in predictions:
        errs.append(np.sum(np.abs(prediction - true)))
    predictor_names = ["Euler predictor", "Inverse Euler\n predictor", "Trapezoidal predictor",
     "Runge Kutta\n 4-degree predictor", "PE(CE)^2 predictor with:\n predictor = Euler,\n corrector = Inverse Euler",
      "PECE predictor with:\n predictor = Euler,\n corrector = Trapezoidal"]
    plt.title('Task {}, errors, T = {}'.format(task_num, integration_period))
    plt.bar(np.arange(len(predictor_names)), height=errs)
    plt.xticks(np.arange(len(predictor_names)), predictor_names)
    plt.show()


def zadatak_1():
    def true_func(integration_period, max_time, initial_state):
        predicted = [initial_state]
        x0_0 = initial_state[0][0]
        x0_1 = initial_state[1][0]
        t = integration_period
        while t <= max_time:
            predicted.append(np.array([[x0_0 * math.cos(t) + x0_1 * math.sin(t)], [x0_1 * math.cos(t) - x0_0 * math.sin(t)]]))
            t += integration_period
        return predicted

    print_task_num(1)
    ls = LinearSystem(np.array([[0, 1], [-1, 0]]), np.array([[1], [1]]))
    s_e = (0.01, 10)
    predictors = init_predictors(*s_e)
    predictions = predict(predictors, ls)
    plot_predictions(predictions, *s_e, 1)
    true = true_func(*s_e, np.array([[1], [1]]))
    plot_errs(true, predictions, *s_e, 1)

def zadatak_2():
    print_task_num(2)
    ls = LinearSystem(np.array([[0, 1], [-200, -102]]), np.array([[1], [-2]]))
    s_e = (0.1, 1)
    predictors = init_predictors(*s_e)
    predictions = predict(predictors, ls)
    plot_predictions(predictions, *s_e, 2)
    integration_period = 0.01
    fig, axes = plt.subplots(1)
    fig.set_size_inches(18, 10)
    fig.suptitle('Task {}, Runge Kutta 4-degree predictor variable predictions, T = {}'.format(2, integration_period))
    rk4 = RungeKutta4Predictor(integration_period, 1)
    preds = rk4.predict(ls)
    time = np.linspace(0, 1, preds.shape[0])
    for j in range(2):
        axes.plot(time, preds[:, j], label="x[{}]".format(j))
    axes.legend()
    plt.show()


def zadatak_3():
    print_task_num(3)
    ls = LinearSystem(np.array([[0, -2], [1, -3]]), np.array([[1], [3]]), np.array([[2, 0], [0, 3]]), lambda x: np.array([[1], [1]]))
    s_e = (0.01, 10)
    predictors = init_predictors(*s_e)
    predictions = predict(predictors, ls)
    plot_predictions(predictions, *s_e, 3)

def zadatak_4():
    print_task_num(4)
    ls = LinearSystem(np.array([[1, -5], [1, -7]]), np.array([[-1], [3]]), np.array([[5, 0], [0, 3]]), lambda x: np.array([[x], [x]]))
    s_e = (0.01, 10)
    predictors = init_predictors(*s_e)
    predictions = predict(predictors, ls)
    plot_predictions(predictions, *s_e, 4)


def main():
    zadatak_1()
    zadatak_2()
    zadatak_3()
    zadatak_4()

if __name__ == "__main__":
    main()
