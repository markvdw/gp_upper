# Copyright 2017 Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import gpflow

float_type = gpflow.settings.dtypes.float_type


class SGPU(gpflow.sgpr.SGPR):
    """
    Upper bound for the GP regression marginal likelihood. Upper bound counterpart to SGPR. The key reference is

    ::

      @misc{titsias_2014,
        title={Variational Inference for Gaussian and Determinantal Point Processes},
        url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
        publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
        author={Titsias, Michalis K.},
        year={2014},
        month={Dec}
      }
    """

    def build_likelihood(self):
        # Upper bound - its negative will be minimized
        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.cast(tf.shape(self.Y)[0], float_type)

        Kdiag = self.kern.Kdiag(self.X)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=float_type) * gpflow.settings.numerics.jitter_level
        Kuf = self.kern.K(self.Z, self.X)

        L = tf.cholesky(Kuu)
        LB = tf.cholesky(Kuu + self.likelihood.variance ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))

        LinvKuf = tf.matrix_triangular_solve(L, Kuf, lower=True)
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(LinvKuf ** 2.0)  # Using the Trace bound, from Titsias' presentation
        # Kff = self.kern.K(self.X)
        # Qff = tf.matmul(Kuf, LinvKuf, transpose_a=True)
        # c = tf.reduce_max(tf.reduce_sum(tf.abs(Kff - Qff), 0))  # Alternative bound on max eigenval
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.log(2 * np.pi * self.likelihood.variance)
        logdet = tf.reduce_sum(tf.log(tf.diag_part(L))) - tf.reduce_sum(tf.log(tf.diag_part(LB)))

        LC = tf.cholesky(Kuu + corrected_noise ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))
        v = tf.matrix_triangular_solve(LC, corrected_noise ** -1.0 * tf.matmul(Kuf, self.Y), lower=True)
        quad = -0.5 * corrected_noise ** -1.0 * tf.reduce_sum(self.Y ** 2.0) + 0.5 * tf.reduce_sum(v ** 2.0)

        return -(const + logdet + quad)  # Return negative upper bound, so upper bound will be minimised

    def build_predict(self, Xnew, full_cov=False):
        raise NotImplementedError

    def build_prior(self):
        return 0.0

    @gpflow.param.AutoFlow()
    def compute_upper_bound(self):
        return -self.build_likelihood()
