"""This module contains all custom kernels for the GP"""
import gpflow
import tensorflow as tf


class Delta_Kernel(gpflow.kernels.Kern):
    """Delta kernel"""

    def __init__(self, dim_problem):

        gpflow.kernels.Kern.__init__(
            self, input_dim=1, active_dims=[dim_problem])

    def K(self, X, X2=None, presliced=False):
        """Caluclate Gram matrix"""

        if X2 is None:
            X2 = X

        if not presliced:
            X, X2 = self._slice(X, X2)

        return tf.multiply(X, tf.transpose(X2))

    def Kdiag(self, X, presliced=False):
        """Calculate only diagonal entries"""

        if not presliced:
            X, _ = self._slice(X, None)

        return tf.reshape(X, (-1,))
