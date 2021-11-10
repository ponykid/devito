import numpy as np
from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
from devito import (Grid, Dimension, Function, TimeFunction, Eq, Inc,
                    Operator, norm)  # noqa
from devito.types import Scalar
from devito.tools import as_list
from examples.seismic import TimeAxis, RickerSource
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

shape = (21, 21)
extent = (100, 100)
origin = (0., 0.)

v = np.empty(shape, dtype=np.float32)
v[:, :11] = 1.5
v[:, 11:] = 2.5

grid = Grid(shape=shape, extent=extent, origin=origin)
x, y = grid.dimensions
time = grid.time_dim
t = grid.stepping_dim
t0 = 0.
tn = 1000.
dt = 1.6
time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.010
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=2, time_range=time_range)

domain_size = np.array(extent)

# Setup sources
src.coordinates.data[0, :] = domain_size*.175
src.coordinates.data[0, -1] = 11
src.coordinates.data[1, :] = domain_size*.545
src.coordinates.data[1, -1] = 41


u = TimeFunction(name="u", grid=grid, space_order=2)
m = Function(name='m', grid=grid)
m.data[:] = 1./(v*v)

# Injection to field u.forward
src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
op = Operator(src_term)
op(time=time_range.num-1, dt=dt)
print(norm(u))
norm_ref = norm(u)
u2 = u

# Get the nonzero indices to nzinds tuple
nzinds = np.nonzero(u.data[0])
assert len(nzinds) == len(shape)

# Create source mask and source id
s_mask = Function(name='s_mask', shape=grid.shape,
                       dimensions=grid.dimensions, dtype=np.int32)
source_id = Function(name='source_id', shape=grid.shape,
                     dimensions=grid.dimensions, dtype=np.int32)

s_mask.data[nzinds[0], nzinds[1]] = 1
source_id.data[nzinds[0], nzinds[1]] = tuple(np.arange(len(nzinds[0])))

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(s_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(u.data[0])))

# Create nnz_mask
nnz_mask = Function(name='nnz_mask', shape=as_list(grid.shape[0], ),
                    dimensions=(grid.dimensions[0], ), dtype=np.int32)

nnz_mask.data[:] = s_mask.data[:, :].sum(1)
assert len(nnz_mask.dimensions) == 1

id_dim = Dimension(name='id_dim')
save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], source_id],
                           expr=src_term.expr)

op1 = Operator([save_src_term])
op1.apply(time=time_range.num-1, dt=dt)

u.data[:] = 0

maxz = len(np.unique(nzinds[1]))
sparse_shape = as_list((grid.shape[0], maxz))  # Change only 2nd dim

sp_yi = Dimension(name='sp_yi')
sp_sm = Function(name='sp_sm', shape=sparse_shape, dimensions=(x, sp_yi),
                 dtype=np.int32)

# Now holds IDs
sp_sm.data[nzinds[0], :] = tuple(nzinds[1][:len(np.unique(nzinds[1]))])

assert(np.count_nonzero(sp_sm.data) == len(nzinds[0]))
assert(len(sp_sm.dimensions) == 2)

yind = Scalar(name='yind', dtype=np.int32)

eq0 = Eq(sp_yi.symbolic_max, nnz_mask[x] - 1, implicit_dims=(time, x))
# eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
# myexpr = save_src[time, source_id[x, sp_sm[x, sp_yi]]]
eq1 = Inc(u.forward[t, x, sp_sm[x, sp_yi]], save_src[time, source_id[x, sp_sm[x, sp_yi]]])

src_term2 = [eq0, eq1]

op2 = Operator(src_term2)
print("===Temporal blocking======================================")
op2.apply(time=time_range.num - 1, dt=dt)
print(norm(u))
norm_sol = norm(u)

print(norm_ref)
print(norm_sol)

assert np.isclose(norm_ref, norm_sol)
assert (u.data[0, :].all() == u2.data[0, :].all())

# plt.figure()
# plt.plot(save_src.data[:, 0]); pause(1)
# plt.plot(src.data[:, 0]); pause(1)
# plt.plot(src.data[:, 1]); pause(1)

import pdb;pdb.set_trace()

# plt.plot(save_src); pause(1)
