#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import numpy.linalg as lin

nn = 256 # size
mm = 10 # number of eigenvectors


print("diff_0")
diff_0 = np.eye(nn - 1, nn, k=1) - np.eye(nn - 1, nn, k=0)
print(diff_0)

print("diff_1")
diff_1 = np.eye(nn, nn - 1, k=0) - np.eye(nn, nn - 1, k=-1)
#diff_1[0,0] = 0
#diff_1[-1,-1] = 0
print(diff_1)


xxs = np.linspace(-1, 1, nn)
dx = xxs[1] - xxs[0]
assert np.abs(xxs[1:] - xxs[:-1] - dx).max() < 1e-5 # dx is constant

print("lap_comb")
lap_comb =  -1 * diff_1 @ diff_0 / dx / dx
print(lap_comb)
print("eigh(lap_comb)")
lap_comb_eigvals, lap_comb_eigvectors = lin.eigh(lap_comb)
#print(lap_comb_eigvals)
assert np.abs(lap_comb_eigvals - lin.eigvalsh(lap_comb)).max() < 1e-5 # eigh <->eigvalsh 

print("heat_1x")
alpha = .4
heat_1x = np.eye(nn, nn) - alpha * dx * dx * lap_comb 
print(heat_1x)
print("eigh(heat_1x)")
heat_1x_eigvals, heat_1x_eigvectors = lin.eigh(heat_1x)
assert np.abs(heat_1x - heat_1x_eigvectors @ np.diag(heat_1x_eigvals) @ heat_1x_eigvectors.T).max() < 1e-5 # check decomposition

def make_heat_kkx(kk):
    return heat_1x_eigvectors @ np.diag(np.power(heat_1x_eigvals, kk)) @ heat_1x_eigvectors.T

print("yys")
yys = np.zeros(nn)
yys[nn//3:2*nn//3] = 1
print(yys)

from pylab import *

figure()
title("$\Delta$ spectrum")
xlabel("rank [na]")
ylabel("energy // eigenvalue * dx [1/dx]")
plot(lap_comb_eigvals * dx)
axvline(mm, ls=":", color="k")

figure()
heat_current = yys.copy()
heat_10x = np.eye(nn, nn)
for kk in range(10):
    plot(heat_current, label="t={}*dt".format(kk))
    heat_current = heat_1x @ heat_current
    heat_10x = heat_1x @ heat_10x
zzs = make_heat_kkx(10) @ yys 
plot(zzs, label="t=10*dt !!")
print("10x_error", np.abs(zzs - heat_current).max())
legend()

figure()
heat_current = yys.copy()
heat_100x = np.eye(nn, nn)
for kk in range(10):
    plot(heat_current, label="t={}*dt".format(10 * kk))
    heat_current = heat_10x @ heat_current
    heat_100x = heat_10x @ heat_100x
zzs = make_heat_kkx(100) @ yys 
plot(zzs, label="t=100*dt !!")
print("100x_error", np.abs(zzs - heat_current).max())
legend()

figure()
heat_current = yys.copy()
for kk in range(10):
    plot(heat_current, label="t={}*dt".format(100 * kk))
    heat_current = heat_100x @ heat_current
zzs = make_heat_kkx(1000) @ yys 
plot(zzs, label="t=1000*dt !!")
print("100x_error", np.abs(zzs - heat_current).max())
legend()

show()
