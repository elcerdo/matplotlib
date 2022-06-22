#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import numpy.linalg as lin

nn = 256 # size
mm = 16 # number of eigenvectors

tts = np.linspace(0, 1, nn)

dt = tts[1] - tts[0]
assert np.abs(tts[1:] - tts[:-1] - dt).max() < 1e-5 # dt is constant

print("diff_0")
diff_0 = np.eye(nn - 1, nn, k=1) - np.eye(nn - 1, nn, k=0)
print(diff_0)

print("diff_1")
diff_1 = np.eye(nn, nn - 1, k=0) - np.eye(nn, nn - 1, k=-1)
#diff_1[0,0] = 0
#diff_1[-1,-1] = 0
print(diff_1)

print("lap_comb")
lap_comb =  -1 * diff_1 @ diff_0 / dt / dt
print(lap_comb)
print("eigvalsh(lap_comb)")
lap_comb_eigvals, lap_comb_eigvectors = lin.eigh(lap_comb)
assert np.abs(lap_comb_eigvals - lin.eigvalsh(lap_comb)).max() < 1e-5 

print("heat")
heat = np.eye(nn, nn) 
print(heat)

from pylab import *

figure()
xlabel("rank [na]")

plot(lap_comb_eigvals)

show()
