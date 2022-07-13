
import os.path
import numpy as np
import numpy.linalg as lin
from pylab import *

nn = 32
mm = nn * nn

show_figure = True
save_figure = False

def dump_figure(figpath):
    if not save_figure:
        return
    if figpath is None:
        return
    print("saving \"{}\"".format(figpath))
    savefig(os.path.join("laplace_beltrami_2d_figs", "{}.png".format(figpath)))
    savefig(os.path.join("laplace_beltrami_2d_figs", "{}.pdf".format(figpath)))

def dump(label, value):
    print("{} shape {} min {} max {}".format(
        label,
        value.shape,
        value.min(),
        value.max()))

iis, jjs = np.meshgrid(np.arange(nn, dtype=int), np.arange(nn, dtype=int))
iis = iis.flatten()
jjs = jjs.flatten()
dump("iis", iis)
dump("jjs", jjs)

cell_to_indices = {}
for index, cell in enumerate(zip(iis, jjs)):
    cell_to_indices[cell] = index

squared_distances = (iis - nn // 2) * (iis - nn // 2) + (jjs - nn // 2) * (jjs - nn // 2)
is_insides = np.logical_and(squared_distances < (11 * nn // 32)**2, squared_distances >= (7 * nn // 32)**2)

### compute lap & lap_bel

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
dump("lap", lap)

lap_bel = np.zeros((mm, mm), dtype=float)
for index, cell in enumerate(zip(iis, jjs)):
    accum = 0
    is_inside = is_insides[cell_to_indices[cell]]
    for cell_ in neighboring_cells(cell):
        is_inside_ = is_insides[cell_to_indices[cell_]]
        weight = 1. if is_inside_ == is_inside else 1e-4
        accum += weight
        index_ = cell_to_indices[cell_]
        lap_bel[index, index_] -= weight
    lap_bel[index, index] = accum
lap_bel += 1e-6 * np.eye(mm)
dump("lap_bel", lap_bel)

### eigen decomposion of lap

print("eigh(lap)")
lap_eigvalues, lap_eigvectors = lin.eigh(lap)
print(lap_eigvalues)
assert np.abs(lap_eigvectors @ np.diag(lap_eigvalues) @ lap_eigvectors.T - lap).max() < 1e-5
assert np.abs(lap_eigvectors @ np.diag(1 / lap_eigvalues) @ lap_eigvectors.T @ lap - np.eye(mm)).max() < 1e-5
assert np.abs(lap @ lap_eigvectors @ np.diag(1 / lap_eigvalues) @ lap_eigvectors.T - np.eye(mm)).max() < 1e-5
assert np.abs(lap_eigvectors @ lap_eigvectors.T - np.eye(mm)).max() < 1e-5

### eigen decomposion of lap

print("eigh(lap_bel)")
lap_bel_eigvalues, lap_bel_eigvectors = lin.eigh(lap_bel)
print(lap_bel_eigvalues)
assert np.abs(lap_bel_eigvectors @ np.diag(lap_bel_eigvalues) @ lap_bel_eigvectors.T - lap_bel).max() < 1e-5
assert np.abs(lap_bel_eigvectors @ np.diag(1 / lap_bel_eigvalues) @ lap_bel_eigvectors.T @ lap_bel - np.eye(mm)).max() < 1e-5
assert np.abs(lap_bel @ lap_bel_eigvectors @ np.diag(1 / lap_bel_eigvalues) @ lap_bel_eigvectors.T - np.eye(mm)).max() < 1e-5
assert np.abs(lap_bel_eigvectors @ lap_bel_eigvectors.T - np.eye(mm)).max() < 1e-5

### figures

figure()
title("is_insides")
imshow(is_insides.reshape(nn, nn))

figure()
title("2D Laplace spectrum")
xlabel("rank")
ylabel("$\lambda_i$")
plot(lap_eigvalues)
dump_figure("lap_spectrum")

figure()
title("2D Laplace-Beltrami spectrum")
xlabel("rank")
ylabel("$\lambda_i$")
plot(lap_bel_eigvalues)
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
    label="2D Laplace eigen vectors",
    figpath="lap_eigenvectors")

display_eigvectors(
    lap_bel_eigvalues,
    lap_bel_eigvectors, [
    0, 1, 2, 3, 4,
    5, 6, 7, 7, 8,
    33, 34, 35, 58, 59], 3, 5,
    label="2D Laplace-Beltrami eigen vectors $\phi_i$",
    figpath="lap_eigenvectors")

if show_figure:
    show()
