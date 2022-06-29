#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import numpy.linalg as lin
from pylab import *

nn = 32

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
lap += 1e-5 * np.eye(mm)

print(lap)
print(lap.sum(axis=0))
print(lap.sum(axis=1))
print(np.abs(lap - lap.T).max())
print(lin.matrix_rank(lap))
assert np.abs(lap - lap.T).max() < 1e-5

### eigen decomposion of lap

lap_eigvalues, lap_eigvectors = lin.eigh(lap)
print(lap_eigvalues)
assert np.abs(lap_eigvectors @ np.diag(lap_eigvalues) @ lap_eigvectors.T - lap).max() < 1e-5
assert np.abs(lap_eigvectors @ np.diag(1 / lap_eigvalues) @ lap_eigvectors.T @ lap - np.eye(mm)).max() < 1e-5
assert np.abs(lap @ lap_eigvectors @ np.diag(1 / lap_eigvalues) @ lap_eigvectors.T - np.eye(mm)).max() < 1e-5

figure()
title("2D $\Delta$ spectrum")
xlabel("rank")
ylabel("$\lambda_i$")
plot(lap_eigvalues)

def display_eigvectors(ranks, foo):
    extensions = {
        0: "st",
        1: "nd",
        2: "rd",
    }
    figure()
    title("2D eigen vectors")
    for kk, rank in enumerate(ranks):
        ll = lap_eigvalues[rank]
        vv = lap_eigvectors[:, rank].reshape(nn, nn)
        subplot(foo, foo, kk + 1)
        axis('off')
        title("{}{} $\lambda_{{{}}} = {:.2f}$".format(
            rank + 1,
            extensions.get(rank % 10, "th"),
            rank,
            ll))
        ss = 1 / 4 * 8 / nn
        imshow(vv, vmin=-ss, vmax=ss)
    #colorbar()

display_eigvectors([
    0, 1, 2, 3,
    4, 5, 6, 7,
    32, 33, 34, 35,
    56, 57, 58, 59], 4)

### heat diffusion

def make_heat_ope(dt, nstep):
    heat = np.eye(mm, mm)
    dheat = np.eye(mm, mm) + dt * lap / nstep
    for kk in range(nstep):
        heat = heat @ dheat
    return heat

### eigen decomposion of heat

theat = 10
heat = make_heat_ope(theat, 1)
heat_eigvalues, heat_eigvectors = lin.eigh(heat)

figure()
title("2D $heat = I + t \Delta$ spectrum {}s".format(theat))
xlabel("rank")
ylabel("$\lambda_i$")
plot(heat_eigvalues, label="$\lambda_{heat}$")
plot(lap_eigvalues * theat + 1, label="$1+t\lambda_{\Delta}$")
legend()

assert np.abs(theat * lap_eigvalues + 1 - heat_eigvalues).max() < 1e-5

### heat solve

heat_input = np.zeros((nn, nn))
heat_input[nn // 3 : 2 * nn // 3,  nn // 2 : 3 * nn // 4] = 1
heat_input = heat_input.flatten()



def display_heat_solutions(heat_profiles_labels, label=None):
    figure()
    if label is not None:
        title(label)
    foo = len(heat_profiles_labels)
    current = 1
    for heat_profile, heat_label in heat_profiles_labels:
        subplot(3, foo, current)
        title(heat_label)
        current += 1
        axis('off')
        imshow(heat_profile.reshape(nn, nn), vmin=0, vmax=1)
        axhline(nn / 2, color="r")
    for heat_profile, heat_label in heat_profiles_labels:
        subplot(3, foo, current)
        current += 1
        axis('off')
        imshow(np.log(heat_profile.reshape(nn, nn) + 1e-9), cmap=plt.get_cmap("flag"), vmin=-8, vmax=0)
    for heat_profile, heat_label in heat_profiles_labels:
        subplot(3, foo, current)
        current += 1
        axis('off')
        ylim(-.1, 1.1)
        plot(heat_profile.reshape(nn, nn)[nn // 2, :])

display_heat_solutions([
    (heat_input, "input"),
    (lin.solve(make_heat_ope(1, 1), heat_input), "1s"),
    (lin.solve(make_heat_ope(10, 1), heat_input), "10s"),
    (lin.solve(make_heat_ope(100, 1), heat_input), "100s"),
    (lin.solve(make_heat_ope(1000, 1), heat_input), "1000s"),
], "Euler heat solutions")




show()

