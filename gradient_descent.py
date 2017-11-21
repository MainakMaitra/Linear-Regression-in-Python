import copy

import numpy as np
import sklearn


class GradientDescent:

    def __init__(self, parameters, hypothesis):
        self.parameters = copy.copy(parameters)
        self.hypothesis = hypothesis
        return

    def get_epochs_number(self):
        return self.parameters.n_epochs

    def get_regularization_parameter(self):
        return self.parameters.l2

    def get_learning_rate(self):
        return self.parameters.learning_rate

    def get_batch_size(self):
        return self.parameters.batch_size

    def update_learning_rate(self):
        self.parameters.learning_rate *= self.parameters.decay

    def regularize(self, derivatives, coefs):
        derivatives[1:] += coefs[1:] * self.get_regularization_parameter()

    def shuffle_data_set(self, data_set):
        if self.parameters.shuffle:
            return sklearn.utils.shuffle(*data_set)
        else:
            return data_set

    def batch(self, data_set, coefs):
        x, y = data_set
        hypothetical_x = np.apply_along_axis(self.hypothesis, 1, x, coefs)
        batch_sum = hypothetical_x - y
        derivatives = batch_sum.dot(x)
        self.regularize(derivatives, coefs)
        coefs -= derivatives * self.get_learning_rate() / x.shape[0]

    def epoch(self, data_set, coefs):
        n_examples = data_set[0].shape[0]
        batch_size = self.get_batch_size()
        data_set = self.shuffle_data_set(data_set)

        for i in range(0, n_examples, batch_size):
            begin, end = i, min(i+batch_size, n_examples)
            self.batch((data_set[0][begin:end], data_set[1][begin:end]), coefs)

        self.update_learning_rate()

    def descent(self, data_set):
        n_dim = data_set[0].shape[1]
        coefs = np.zeros(n_dim)

        for _ in range(0, self.get_epochs_number()):
            self.epoch(data_set, coefs)

        return coefs
