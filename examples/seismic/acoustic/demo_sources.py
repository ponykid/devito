import numpy as np
from matplotlib.pyplot import pause # noqa
from devito import Grid, Operator, norm
from aligner import aligner
from examples.seismic import TimeAxis, RickerSource
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

shape = (2100, 2100)
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
src.coordinates.data[1, -1] = 11

src_term2, u = aligner(grid, v, src, dt, time_range)
import pdb;pdb.set_trace()
op2 = Operator(src_term2)

print("===Temporal blocking======================================")
op2.apply(time=time_range.num - 1)

norm_sol = norm(u)
print(norm_sol)

# assert np.isclose(norm_ref, norm_sol)
# assert (u.data[0, :].all() == u2.data[0, :].all())

# plt.figure()
# plt.plot(save_src.data[:, 0]); pause(1)
# plt.plot(src.data[:, 0]); pause(1)
# plt.plot(src.data[:, 1]); pause(1)

# plt.plot(save_src); pause(1)
