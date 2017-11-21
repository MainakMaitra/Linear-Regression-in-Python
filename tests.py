import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from parameters import Parameters
from gradient_descent import GradientDescent


def linear_regression_hypothesis(x, coefs):
    return np.dot(x, coefs)


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_updating_learning(self):
        parameters = Parameters(learning_rate=10, decay=0.5)
        gd = GradientDescent(parameters, linear_regression_hypothesis)

        self.assertEqual(gd.get_learning_rate(), 10)
        gd.update_learning_rate()
        self.assertEqual(gd.get_learning_rate(), 5)

    def test_regularization(self):
        derivatives = np.zeros(10)
        coefs = np.ones(10)
        parameters = Parameters(l2=2)
        derivatives_expected = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2])

        gd = GradientDescent(parameters, linear_regression_hypothesis)
        gd.regularize(derivatives, coefs)

        np.testing.assert_array_equal(derivatives, derivatives_expected)

    def test_shuffling(self):
        x = np.array(range(1000))
        x_not_changed = x.copy()
        y = x.copy()
        parameters = Parameters(shuffle=True)

        gd = GradientDescent(parameters, linear_regression_hypothesis)
        x, y = gd.shuffle_data_set((x, y))

        self.assertTrue(np.any(np.not_equal(x, x_not_changed)))
        np.testing.assert_array_equal(x, y)

    def test_batch(self):
        gd = GradientDescent(Parameters(), linear_regression_hypothesis)
        coefs = np.ones(2)
        x = np.array([[1, i] for i in range(10)])
        y = np.array([2*i + 10 for i in range(10)])

        gd.batch((x, y), coefs)
        self.assertTrue(np.all(np.greater(coefs, 1)))

    def test_epoch(self):
        x = np.array([[1, i] for i in range(10)])
        y = np.array([2*i + 10 for i in range(10)])
        gd = GradientDescent(Parameters(batch_size=4, learning_rate=1, decay=0.5), linear_regression_hypothesis)
        gd.epoch((x, y), np.zeros(2))
        self.assertEqual(gd.parameters.learning_rate, 0.5)

    def test_descent_easy(self):
        x = np.array([[1, i] for i in range(1000)])
        y = np.array([10 + 2*i for i in range(1000)])
        parameters = Parameters(batch_size=100, shuffle=True)
        gd = GradientDescent(parameters, linear_regression_hypothesis)

        xscal = StandardScaler(copy=False)
        x = xscal.fit_transform(x)
        yscal = StandardScaler(copy=False)
        y = yscal.fit_transform(y)

        coef0, coef1 = gd.descent((x, y))
        self.assertAlmostEqual(coef0, 0)
        self.assertAlmostEqual(coef1, 1)
