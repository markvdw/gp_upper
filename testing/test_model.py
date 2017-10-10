# Copyright 2016 Mark van der Wilk.
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
# limitations under the License.from __future__ import print_function

from __future__ import print_function

import unittest

import numpy as np

import gp_upper
import gpflow


def get_parameter_dict(model):
    return {p.full_name.split(model.full_name)[1]: p.read_value() for p in model.parameters}


def set_parameter_dict(model, d):
    assign_params = {p.full_name.split(model.full_name)[1]: p for p in model.parameters}
    [assign_params[k].assign(d[k]) for k in d]


class TestUpperBound(unittest.TestCase):
    """
    Test for SGPU
    """

    def setUp(self):
        self.X = np.random.rand(100, 1)
        self.Y = np.sin(1.5 * 2 * np.pi * self.X) + np.random.randn(*self.X.shape) * 0.1

    def test_few_inducing_points(self):
        vfe = gpflow.models.SGPR(self.X, self.Y, gpflow.kernels.RBF(1), self.X[:10, :].copy())
        vfe.compile()
        gpflow.train.ScipyOptimizer().minimize(vfe)

        upper = gp_upper.SGPU(self.X, self.Y, gpflow.kernels.RBF(1), self.X[:10, :].copy())
        upper.kern.set_trainable(False)
        upper.likelihood.set_trainable(False)
        upper.compile()
        set_parameter_dict(upper, get_parameter_dict(vfe))
        gpflow.train.ScipyOptimizer().minimize(upper)

        full = gpflow.models.GPR(self.X, self.Y, gpflow.kernels.RBF(1))
        full.kern.lengthscales = vfe.kern.lengthscales.read_value()
        full.kern.variance = vfe.kern.variance.read_value()
        full.likelihood.variance = vfe.likelihood.variance.read_value()
        full.compile()

        lml_upper = upper.compute_upper_bound()
        lml_vfe = vfe.compute_log_likelihood()
        lml_full = full.compute_log_likelihood()

        self.assertTrue(lml_upper > lml_full > lml_vfe)


if __name__ == "__main__":
    unittest.main()
