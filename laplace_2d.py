#!/usr/bin/env python3
# coding: utf-8

import os.path
import numpy as np
import numpy.linalg as lin
from pylab import *

nn = 64
mm = nn * nn

show_figure = False
save_figure = True

plt.rcParams['figure.figsize'] = [14 / 2, 9 / 2]

def dump_figure(figpath):
    if not save_figure:
        return
    if figpath is None:
        return
    print("saving \"{}\"".format(figpath))
    savefig(os.path.join("laplace_2d_figs", "{}.png".format(figpath)))
    savefig(os.path.join("laplace_2d_figs", "{}.pdf".format(figpath)))

def dump(label, value):
    print("{} shape {} min {} max {}".format(
        label,
        value.shape,
        value.min(),
        value.max()))

### build complex

iis, jjs = np.meshgrid(np.arange(nn, dtype=int), np.arange(nn, dtype=int))

iis = iis.flatten()
jjs = jjs.flatten()

dump("iis", iis)
dump("jjs", jjs)

assert mm == iis.size

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
lap += 1e-6 * np.eye(mm)

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
assert np.abs(lap_eigvectors @ lap_eigvectors.T - np.eye(mm)).max() < 1e-5

figure()
title("2D $\Delta$ spectrum")
xlabel("rank")
ylabel("$\lambda_i$")
plot(lap_eigvalues)
dump_figure("lap_spectrum")

def display_eigvectors(eigvalues, eigvectors, ranks, foo, bar, label=None, figpath=None):
    extensions = {
        0: "st",
        1: "nd",
        2: "rd",
    }
    figure()
    if label is not None:
        suptitle(label)
    for kk, rank in enumerate(ranks):
        ll = eigvalues[rank]
        vv = eigvectors[:, rank].reshape(nn, nn)
        subplot(foo, bar, kk + 1)
        axis('off')
        rank_str = "{}{}".format(rank + 1, extensions.get(rank % 10, "th"))
        title("$\lambda_{{{}}} = {:.2f}$".format(
            rank,
            ll))
        ss = 1 / 4 * 8 / nn
        imshow(vv, vmin=-ss, vmax=ss)
    #colorbar()
    dump_figure(figpath)

display_eigvectors(
    lap_eigvalues,
    lap_eigvectors, [
    0, 1, 2, 3, 4,
    5, 6, 7, 7, 8,
    33, 34, 35, 58, 59], 3, 5,
    label="2D $\Delta$ eigen vectors $\phi_i$",
    figpath="lap_eigenvectors")

### heat diffusion

def make_heat_ope(dt, nstep):
    heat = np.eye(mm, mm)
    dheat = np.eye(mm, mm) + dt * lap / nstep
    for kk in range(nstep):
        heat = heat @ dheat
    return heat

### heat solve

heat_input = np.zeros((nn, nn))
heat_input[nn // 3 : 2 * nn // 3,  nn // 2 : 3 * nn // 4] = 1
heat_input = heat_input.flatten()

heat_input_ = 8 * np.eye(nn,k=nn // 3)
heat_input_ = heat_input_.flatten()

def display_heat_solutions(heat_profiles_labels, label=None, figpath=None):
    figure()
    if label is not None:
        suptitle(label)
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
    dump_figure(figpath)

display_heat_solutions([
    (heat_input, "input"),
    (lin.solve(make_heat_ope(1, 1), heat_input), "1s"),
    (lin.solve(make_heat_ope(10, 1), heat_input), "10s"),
    (lin.solve(make_heat_ope(100, 1), heat_input), "100s"),
    (lin.solve(make_heat_ope(1000, 1), heat_input), "1000s"),
], label="Euler $heat$ solutions", figpath="heat_euler_00")

display_heat_solutions([
    (heat_input_, "input"),
    (lin.solve(make_heat_ope(1, 1), heat_input_), "1s"),
    (lin.solve(make_heat_ope(10, 1), heat_input_), "10s"),
    (lin.solve(make_heat_ope(100, 1), heat_input_), "100s"),
    (lin.solve(make_heat_ope(1000, 1), heat_input_), "1000s"),
], label="Euler $heat$ solutions", figpath="heat_euler_01")

### eigen decomposion of heat

theat = 50.
heat = make_heat_ope(theat, 1)
heat_eigvalues, heat_eigvectors = lin.eigh(heat)
assert np.abs(theat * lap_eigvalues + 1 - heat_eigvalues).max() < 1e-5

display_eigvectors(
    heat_eigvalues,
    heat_eigvectors, [
    0, 1, 2, 3, 4,
    5, 6, 7, 7, 8,
    33, 34, 35, 58, 59], 3, 5,
    label="2D $heat$ eigen vectors $\phi_i$",
    figpath="heat_eigenvectors")

figure()
title("2D $heat = I + t \Delta$ spectrum {}s".format(theat))
xlabel("rank")
ylabel("$\lambda_i$")
plot(heat_eigvalues, label="$\lambda_{heat}$")
plot(lap_eigvalues * theat + 1, label="$1+t\lambda_{\Delta}$")
legend()
dump_figure("heat_spectrum")

heat_ = np.eye(mm) + theat * lap_eigvectors @ diag(lap_eigvalues) @ lap_eigvectors.T
assert np.abs(heat - heat_).max() < 1e-5

lap_inv = lap_eigvectors @ np.diag(1 / lap_eigvalues) @ lap_eigvectors.T
assert np.abs(lap_inv @ lap - np.eye(mm)).max() < 1e-5

heat_inv = heat_eigvectors @ np.diag(1 / heat_eigvalues) @ heat_eigvectors.T
assert np.abs(heat_inv @ heat - np.eye(mm)).max() < 1e-5

heat_inv_ = lap_eigvectors @ np.diag(1 / (lap_eigvalues * theat + 1)) @ lap_eigvectors.T
assert np.abs(heat_inv_ @ heat - np.eye(mm)).max() < 1e-5

def make_restricted_heat_inv(mm_):
    heat_restrict_eigvectors = heat_eigvectors[:, :mm_]
    heat_restrict_inv = heat_restrict_eigvectors @ np.diag(1 / heat_eigvalues[:mm_]) @ heat_restrict_eigvectors.T
    return heat_restrict_inv

display_heat_solutions([
    (lin.solve(heat, heat_input), "Euler"),
    (make_restricted_heat_inv(mm // 2) @ heat_input, "50%"),
    (make_restricted_heat_inv(mm // 4) @ heat_input, "25%"),
    (make_restricted_heat_inv(mm // 10) @ heat_input, "10%"),
    (make_restricted_heat_inv(mm // 20) @ heat_input, "5%"),
    (make_restricted_heat_inv(mm // 40) @ heat_input, "2.5%"),
    (make_restricted_heat_inv(mm // 100) @ heat_input, "1%"),
],
    label="Restricted eigenbasis $heat$ solutions {}s".format(theat),
    figpath="heat_restricted_00")

display_heat_solutions([
    (lin.solve(heat, heat_input_), "Euler"),
    (make_restricted_heat_inv(mm // 2) @ heat_input_, "50%"),
    (make_restricted_heat_inv(mm // 4) @ heat_input_, "25%"),
    (make_restricted_heat_inv(mm // 10) @ heat_input_, "10%"),
    (make_restricted_heat_inv(mm // 20) @ heat_input_, "5%"),
    (make_restricted_heat_inv(mm // 40) @ heat_input_, "2.5%"),
    (make_restricted_heat_inv(mm // 100) @ heat_input_, "1%"),
],
    label="Restricted eigenbasis $heat$ solutions {}s".format(theat),
    figpath="heat_restricted_01")

print("====================")

### project sdf on eigbasis

xxs = jjs.astype(float) / (nn - 1) * 2 - 1
yys = iis.astype(float) / (nn - 1) * 2 - 1
# xxs -= .5
# yys += .3333
dump("xxs", xxs)
dump("yys", yys)

the = -np.pi / 6
xxs_ = cos(the) * xxs + sin(the) * yys
yys_ = -sin(the) * xxs + cos(the) * yys

def make_restricted_heat_exp(mm_, tt):
    heat_restrict_eigvectors = heat_eigvectors[:, :mm_]
    heat_restrict_inv = heat_restrict_eigvectors @ np.diag(np.exp(-tt * heat_eigvalues[:mm_])) @ heat_restrict_eigvectors.T
    return heat_restrict_inv

def display_opes(opes, examples, label=None, figpath=None):
    figure()
    if label is not None:
        suptitle(label)
    current = 0
    first = True
    for inputs in examples:
        vmin = inputs.min()
        vmax = inputs.max()
        for ope_func, ope_label in opes:
            current += 1
            subplot(len(examples), len(opes), current)
            if first:
                title(ope_label)
            axis('off')
            outputs = ope_func @ inputs
            imshow(outputs.reshape(nn, nn), vmin=vmin, vmax=vmax, cmap=plt.get_cmap("nipy_spectral"))
        first = False
    dump_figure(figpath)

sdf_example = np.sqrt(np.power(xxs - .5, 2) + np.power(yys + .333, 2))
sdf_example_ = np.abs(xxs - .5) + np.abs(yys + .333)
sdf_example__ = np.maximum(np.abs(xxs_ + .5), np.abs(yys_ - .2))
sdf_examples = [
    sdf_example,
    sdf_example_,
    sdf_example__,
    np.minimum(sdf_example__, sdf_example),
]
for kk, ex in enumerate(sdf_examples):
    dump("sdf_example_{:02d}".format(kk), ex)

display_opes([
        (np.eye(mm), "input"),
        (make_restricted_heat_exp(mm // 2, 0), "50%"),
        (make_restricted_heat_exp(mm // 4, 0), "25%"),
        (make_restricted_heat_exp(mm // 10, 0), "10%"),
        (make_restricted_heat_exp(mm // 20, 0), "5%"),
        (make_restricted_heat_exp(mm // 40, 0), "2.5%"),
        (make_restricted_heat_exp(mm // 100, 0), "1%"),
], sdf_examples, label="SDF project on eigen subspaces", figpath="sdf_restricted")

if show_figure:
    show()

