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

import gpflow
import gp_upper


class TestUpperBound(unittest.TestCase):
    """
    Test for SGPU
    """

    def setUp(self):
        self.X = np.random.rand(100, 1)
        self.Y = np.sin(1.5 * 2 * np.pi * self.X) + np.random.randn(*self.X.shape) * 0.1

    def test_few_inducing_points(self):
        vfe = gpflow.sgpr.SGPR(self.X, self.Y, gpflow.kernels.RBF(1), self.X[:10, :].copy())
        vfe.optimize()

        upper = gp_upper.SGPU(self.X, self.Y, gpflow.kernels.RBF(1), self.X[:10, :].copy())
        upper.set_parameter_dict(vfe.get_parameter_dict())
        upper.kern.fixed = True
        upper.likelihood.fixed = True
        upper.optimize()

        full = gpflow.gpr.GPR(self.X, self.Y, gpflow.kernels.RBF(1))
        full.kern.lengthscales = vfe.kern.lengthscales.value
        full.kern.variance = vfe.kern.variance.value
        full.likelihood.variance = vfe.likelihood.variance.value
        full._compile()

        lml_upper = upper.compute_upper_bound()
        lml_vfe = -vfe._objective(vfe.get_free_state())[0]
        lml_full = -full._objective(full.get_free_state())[0]

        self.assertTrue(lml_upper > lml_full > lml_vfe)


if __name__ == "__main__":
    unittest.main()
