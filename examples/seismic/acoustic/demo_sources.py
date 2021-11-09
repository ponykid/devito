import numpy as np
import pytest

from devito import (Grid, Dimension, Function, TimeFunction, Eq, Inc, solve,
                    Operator, norm, cos)  # noqa
from examples.seismic import TimeAxis, RickerSource, Receiver

shape = (101, 101)
extent = (1000, 1000)
origin = (0., 0.)

v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

grid = Grid(shape=shape, extent=extent, origin=origin)
x, y = grid.dimensions
t0 = 0.
tn = 1000.
dt = 1.6
time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.010
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=1, time_range=time_range)

domain_size = np.array(extent)

src.coordinates.data[0, :] = domain_size*.5
src.coordinates.data[0, -1] = 20.

rec = Receiver(name='rec', grid=grid, npoint=101, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0, domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.

u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
m = Function(name='m', grid=grid)
m.data[:] = 1./(v*v)

pde = m * u.dt2 - u.laplace
stencil = Eq(u.forward, solve(pde, u.forward))

src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
rec_term = rec.interpolate(expr=u.forward)

op = Operator(src_term + rec_term)
op(time=time_range.num-1, dt=dt)

#  Get the nonzero indices
nzinds = np.nonzero(u.data[0])  # nzinds is a tuple
assert len(nzinds) == len(shape)

source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y), space_order=0, dtype=np.float32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y), space_order=0, dtype=np.int32)

source_mask.data[nzinds[0], nzinds[1]] = 1
source_id.data[nzinds[0], nzinds[1]] = tuple(np.arange(len(nzinds[0])))

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(u.data[0])))

nnz_shape = (grid.shape[0], )  # Change only 2nd dim
nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x, ), space_order=0, dtype=np.int32)

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
op1.apply(time=time_range.num-1)

import pdb;pdb.set_trace()




'''
op = Operator([stencil] + src_term + rec_term, opt=('advanced'),
              language='openmp')

op(time=time_range.num-1, dt=dt)

assert np.isclose(norm(rec), 490.55, atol=1e-2, rtol=0)
'''