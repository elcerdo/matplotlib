#!/usr/bin/env python3
# coding: utf-8

import numpy as num

num.set_printoptions(precision=3, suppress=True)

nn = 4
aa = num.random.normal(size=(nn, nn))
print("aa")
print(aa)

bb = aa + aa.T
print("sym(aa)")
print(bb)

lb, pb = num.linalg.eig(bb)
print("eig(sym(aa))", lb)
print(pb)
print(num.abs(num.linalg.inv(pb) - pb.T.conj()).max())
print(num.abs(pb @ num.diag(lb) @ pb.T.conj() - bb).max())

cc = aa - aa.T
print("skew(aa)")
print(cc)

lc, pc = num.linalg.eig(cc)
print("eig(skew(aa))", lc)
print(pc)
print(num.abs(num.linalg.inv(pc) - pc.T.conj()).max())
print(num.abs(pc @ num.diag(lc) @ pc.T.conj() - cc).max())

