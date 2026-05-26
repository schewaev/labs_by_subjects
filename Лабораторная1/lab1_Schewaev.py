# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:31:22 2026

@author: schewaev
"""

import numpy as np

rnd = np.random.default_rng()
a = rnd.random((3, 3))
print('a =\n', a)

a_normilized = (a - np.min(a)) / (np.max(a) - np.min(a))
print('normilzed a =\n', a_normilized)