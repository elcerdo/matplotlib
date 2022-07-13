#!/usr/bin/env python3
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin

nn = 64
ss = 3.
aa = 0.3 # np.pi / 6

sdf_example = lambda pps: np.sqrt(np.power(pps.real - .5, 2) + np.power(pps.imag + .333, 2))
sdf_example_ = lambda pps: np.abs(pps.real - .5) + np.abs(pps.imag + .333)
sdf_example__ = lambda pps: np.maximum(np.abs((pps * exp(1j * aa)).real + .5), np.abs((pps * exp(1j * aa)).imag - .2)) - 1>, ss, nn), np.linspace(-ss, ss, nn))
imshow_pps = imshow_pps[0].flatten() + 1j * imshow_pps[1].flatten()
print("imshow_pps", imshow_pps.shape)

from pylab import *

for example in sdf_examples:
    # positions = np.meshgrid(linspace(-.8 * ss, .8 * ss, 3), linspace(-.8 * ss, .8 * ss, 3))
    # positions = positions[0].flatten() + 1j * positions[1].flatten()
    positions = np.linspace(.1+1j, .2-1j, 3, dtype=complex)
    print("positions", positions.shape)
    print(positions)

    foobar = []
    foobar.append(positions.copy())
    for kk in range(10):
        for row in range(positions.size):
            # print("aa", positions[row])
            accum = 1
            for row_ in range(positions.size):
                if row == row_:
                    continue
                accum *= positions[row] - positions[row_]
            print("accum", accum)
            dp = example(positions[row]) / accum
            print("dp", np.absolute(dp))
            positions[row] -= dp
            # print("bb", positions[row])
        foobar.append(positions.copy())

    foobar = array(foobar)
    print("foobar", foobar.shape)

    ll = imshow_pps.real.min(), imshow_pps.imag.min()
    hh = imshow_pps.real.max(), imshow_pps.imag.max()
    ee = (ll[0] - 1 / nn, hh[0] + 1 / nn, ll[1] - 1 / nn, hh[1] + 1 / nn)

    figure()
    axis('equal')
    # axis('off')

    subplot(1, 3, 1)
    imshow(example(imshow_pps).real.reshape(nn, nn), cmap=plt.get_cmap("seismic"), origin="lower", extent=ee, vmin=-ss, vmax=ss)
    subplot(1, 3, 2)
    imshow(example(imshow_pps).imag.reshape(nn, nn), cmap=plt.get_cmap("seismic"), origin="lower", extent=ee, vmin=-ss, vmax=ss)
    subplot(1, 3, 3)
    imshow(np.absolute(example(imshow_pps).reshape(nn, nn)), cmap=plt.get_cmap("viridis"), origin="lower", extent=ee, vmin=0, vmax=ss)
    plot(foobar.real, foobar.imag, "-o")
    plot(foobar.real[0,:], foobar.imag[0,:], "^")

show()

