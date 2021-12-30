import numpy as np

from devito import TimeFunction, Function, Dimension, Operator
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
from examples.cfd import plot_field
from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa


# Define a physical size
nx, ny, nz = (200, 200, 200)
shape = (nx, ny, nz)
spacing = (10., 10., 10)
origin = (0., 0., 0.)

# Define sspace order
so = 4

# Construct model
model = Model(origin=origin, shape=shape, spacing=spacing, space_order=so)

x, y, z = model.grid.dimensions

t0 = 0  # Simulation starts a t=0
tn = 200  # Simulation last 1 second (1000 ms)
dt = 1  # model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=9, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
stx = 0.125
ste = 0.9
stepx = (ste-stx)/int(np.sqrt(src.npoint))

# Uniform x,y source spread
src.coordinates.data[:, :2] = \
    np.array(np.meshgrid(np.arange(stx, ste,
             stepx), np.arange(stx, ste, stepx))).T.reshape(-1, 2) \
    * np.array(model.domain_size[:1])

src.coordinates.data[:, -1] = 20  # Depth is 20m

# Perform source injection to empty grid f (all data equal to 0)
f = TimeFunction(name="f", grid=model.grid, space_order=so, time_order=2)
src_f = src.inject(field=f.forward, expr=src * dt**2 / model.m)
op_f = Operator(src_f)
op_f.apply()

# Get the nonzero indices (nzinds)
nzinds = np.nonzero(f.data[0])
assert len(nzinds) == len(shape)

# Source ID function to hold unique id for each point affected
s_id = Function(name='s_id', shape=model.grid.shape, dimensions=model.grid.dimensions,
                space_order=0, dtype=np.int32)
s_id.data[nzinds] = tuple(np.arange(len(nzinds[0])))

# Helper dimension to schedule loop of different size
id_dim = Dimension(name='id_dim')

time = model.grid.time_dim
save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(time, id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], s_id],
                           expr=src * dt**2 / model.m)

op1 = Operator(save_src_term)
op1.apply()

# Assert that first, last as well as other indices are as expected
assert(s_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
assert(s_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
assert(s_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
       nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)
assert(np.all(np.nonzero(s_id.data)) == np.all(np.nonzero(f.data[0])))
assert (src.shape[0] == save_src.shape[0])
assert (4*src.shape[1] == save_src.shape[1])
