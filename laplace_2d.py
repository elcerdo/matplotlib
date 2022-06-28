#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import numpy.linalg as lin

nn = 8

### build complex

iis, jjs = np.meshgrid(np.arange(nn, dtype=int), np.arange(nn, dtype=int))

iis = iis.flatten()
jjs = jjs.flatten()

mm = iis.size
assert mm == nn * nn

cell_to_indices = {}
for index, cell in enumerate(zip(iis, jjs)):
    cell_to_indices[cell] = index

### compute lap

def neighboring_cells(cell):
    if cell not in cell_to_indices:
        return
    deltas = [(1,0), (-1,0), (0,1), (0,-1)]
    row, col = cell
    for dr, dc in deltas:
        cell_ = (row + dr, col + dc)
        if cell_ in cell_to_indices:
            yield cell_

lap = np.zeros((mm, mm), dtype=float)
for index, cell in enumerate(zip(iis, jjs)):
    accum = 0
    for cell_ in neighboring_cells(cell):
        accum += 1
        index_ = cell_to_indices[cell_]
        lap[index, index_] -= 1
    lap[index, index] = accum

print(lap)
print(lap.sum(axis=0))
print(lap.sum(axis=1))
print(np.abs(lap - lap.T).max())
assert np.abs(lap - lap.T).max() < 1e-5

### eigen decomposion of lap

eigvalues, eigvectors = lin.eigh(lap)
print(eigvalues)

from pylab import *

figure()
title("2D $\Delta$ spectrum")
xlabel("rank")
ylabel("$\lambda_i$")
plot(eigvalues)

extensions = {
    0: "st",
    1: "nd",
    2: "rd",
}

def display_eigvector(rank):
    ll = eigvalues[rank]
    vv = eigvectors[:, rank].reshape(nn, nn)
    figure()
    title("2D {}{} eigen vector $\lambda_{{{}}} = {:.2f}$".format(
        rank + 1,
        extensions.get(rank, "th"),
        rank,
        ll))
    imshow(vv, vmin=-.25, vmax=.25)
    colorbar()

for rank in range(5):
    display_eigvector(rank)

display_eigvector(32)
display_eigvector(56)



show()

