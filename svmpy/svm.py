import svmpy
import numpy as np
import cvxopt.solvers
import logging
import json

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVMTrainer(object):
    def __init__(self,
                 kernel,
                 c):
        self._kernel = kernel
        self._c = c

    # write the model to a txt file
    @staticmethod
    def write_model(model, model_path="model.txt"):
        json.dump(model, open(model_path, "w"))

    def train(self,
              X,
              y):
        """
        Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_model(X, y, lagrange_multipliers)

    def _gram_matrix(self,
                     X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_model(self,
                         X,
                         y,
                         lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        model = {"kernel": svmpy.Kernel.type, "sigma": svmpy.Kernel.sigma,
                 "bias": bias, "weights": support_multipliers.tolist(),
                 "support_vectors": support_vectors.tolist(),
                 "support_vector_labels": support_vector_labels.tolist()}
        return model

    def _compute_multipliers(self,
                             X,
                             y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)
        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    @staticmethod
    def load_model(model_path="model.txt"):
        # load the model from the json file
        model = json.load(open(model_path))
        # set the initialize model to be linear model
        kernel = svmpy.Kernel.linear()
        if model["kernel"] == "gaussian":
            kernel = svmpy.Kernel.gaussian(model["sigma"])
        weights = np.array(model["weights"])
        support_vectors = np.matrix(model["support_vectors"])
        support_vector_labels = np.matrix(model["support_vector_labels"])
        # construct the predictor
        return SVMPredictor(kernel,
                            model["bias"],
                            weights,
                            support_vectors,
                            support_vector_labels)

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()
