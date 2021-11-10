import numpy as np
import pytest
from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
from devito import (Grid, Dimension, Function, TimeFunction, Eq, Inc,
                    Operator, norm)  # noqa
from devito.types import Scalar
from examples.seismic import TimeAxis, RickerSource
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

shape = (21, 21)
extent = (1000, 1000)
origin = (0., 0.)

v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

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

src.coordinates.data[0, :] = domain_size*.175
src.coordinates.data[0, -1] = 410

src.coordinates.data[1, :] = domain_size*.545
src.coordinates.data[1, -1] = 410

u = TimeFunction(name="u", grid=grid, space_order=2)
m = Function(name='m', grid=grid)
m.data[:] = 1./(v*v)

src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

op = Operator(src_term)
op(time=time_range.num-1, dt=dt)

#  Get the nonzero indices
nzinds = np.nonzero(u.data[0])  # nzinds is a tuple
assert len(nzinds) == len(shape)

source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y), dtype=np.int32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y), dtype=np.int32)

source_mask.data[nzinds[0], nzinds[1]] = 1
source_id.data[nzinds[0], nzinds[1]] = tuple(np.arange(len(nzinds[0])))

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(u.data[0])))

nnz_shape = (grid.shape[0], )  # Change only 2nd dim
nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x, ), dtype=np.int32)

nnz_sp_source_mask.data[:] = source_mask.data[:, :].sum(1)
inds = np.where(source_mask.data == 1.)

maxz = len(np.unique(inds[1]))
sparse_shape = (grid.shape[0], maxz)  # Change only 2nd dim

assert len(nnz_sp_source_mask.dimensions) == 1

id_dim = Dimension(name='id_dim')
b_dim = Dimension(name='b_dim')

save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], source_id], expr=src_term.expr)

op1 = Operator([save_src_term])
op1.apply(time=time_range.num-1, dt=dt)

usol = TimeFunction(name="usol", grid=grid, space_order=2)
sp_yi = Dimension(name='sp_yi')

# import pdb; pdb.set_trace()

sp_sm = Function(name='sp_sm', shape=(list(sparse_shape)), dimensions=(x, sp_yi), space_order=0, dtype=np.int32)

# Now holds IDs
sp_sm.data[inds[0], :] = tuple(inds[1][:len(np.unique(inds[1]))])

assert(np.count_nonzero(sp_sm.data) == len(nzinds[0]))
assert(len(sp_sm.dimensions) == 2)

yind = Scalar(name='yind', dtype=np.int32)

eq0 = Eq(sp_yi.symbolic_max, nnz_sp_source_mask[x] - 1, implicit_dims=(time, x))
eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
myexpr = save_src[time, source_id[x, yind]]

eq2 = Inc(usol.forward[t+1, x, yind], myexpr, implicit_dims=(time, x, sp_yi))

op2 = Operator([eq0, eq1, eq2], opt=('advanced'))
print("===Temporal blocking======================================")
op2.apply(time=time_range.num - 1, dt=dt)

print(norm(u))
print(norm(usol))
assert np.isclose(norm(u), norm(usol))

plt.figure()
plt.plot(save_src.data[:, 0]); pause(1)
plt.plot(src.data[:, 0]); pause(1)
plt.plot(src.data[:, 1]); pause(1)


# plt.plot(save_src); pause(1)

