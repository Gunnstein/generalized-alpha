# -*- coding: utf-8 -*-
import numpy as np
import unittest

from . import *


class TestGeneralizedAlphaMethod(unittest.TestCase):
    def test_free_vib_sdof(self):
        m, c, k = 1.0, 0.0, np.pi**2
        h = 1./100
        t = np.arange(0., 1.0, h)
        p = np.zeros((100,), dtype=float)
        d0 = 1.
        v0 = 0.
        a, v, d = generalized_alpha_method(m, c, k, p, h, d0, v0, 1.0)

        w = np.sqrt(k/m)
        dtrue = d0 * np.cos(w*t) + v0 / w * np.sin(w*t)

        np.testing.assert_almost_equal(d[0], dtrue, decimal=4)
