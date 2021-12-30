import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import argparse

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc, solve, Grid
from devito import Operator, norm
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size
from devito.types.basic import Scalar, Symbol # noqa
from mpl_toolkits.mplot3d import Axes3D # noqa


# Define a physical size
shape = (210, 210, 210)  # Number of grid point (nx, nz)
origin = (0., 0., 0.)
extent = (100, 100, 100)

so = 4
# Initialize v field
v = np.empty(shape, dtype=np.float32)
v[:, :, :40] = 2
v[:, :, 40:] = 1

grid = Grid(shape=shape, extent=extent, origin=origin)
x, y, z = grid.dimensions
time = grid.time_dim
t = grid.stepping_dim


# grid = Grid(shape=shape, extent=extent, origin=origin)
m = Function(name='m', grid=grid)
m.data[:] = 1./(v*v)
# plt.imshow(model.vp.data[10, :, :]) ; pause(1)

t0 = 0  # Simulation starts a t=0
tn = 1000  # Simulation last 1 second (1000 ms)
dt = 1.6  # model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=grid, f0=f0,
                   npoint=2, time_range=time_range)

domain_size = np.array(grid.extent)

# First, position source centrally in all dimensions, then set depth
stx = 0.1
ste = 0.9
stepx = (ste-stx)/int(np.sqrt(src.npoint))

# src.coordinates.data[:, :2] = np.array(np.meshgrid(np.arange(stx, ste, stepx),
# np.arange(stx, ste, stepx))).T.reshape(-1,2)*np.array(model.domain_size[:1])

# Setup sources
src.coordinates.data[0, :] = domain_size*.17
src.coordinates.data[0, -1] = 11
src.coordinates.data[1, :] = domain_size*.55
src.coordinates.data[1, -1] = 11


# f : perform source injection on an empty grid
f = TimeFunction(name="f", grid=grid, space_order=4, time_order=2)
src_f = src.inject(field=f.forward, expr=src * dt**2 / m)
# op_f = Operator([src_f], opt=('advanced', {'openmp': True}))
op_f = Operator([src_f])
op_f.apply(time=time_range.num-1)
normf = norm(f)
print("==========")
print(normf)
print("===========")

# uref : reference solution
# uref = TimeFunction(name="uref", grid=model.grid, space_order=so, time_order=2)
# src_term_ref = src.inject(field=uref.forward, expr=src * dt**2 / model.m)
# pde_ref = model.m * uref.dt2 - uref.laplace + model.damp * uref.dt
# stencil_ref = Eq(uref.forward, solve(pde_ref, uref.forward))

# Get the nonzero indices
nzinds = np.nonzero(f.data[0])  # nzinds is a tuple
print(nzinds)
assert len(nzinds) == len(shape)

s_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z), dtype=np.float32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), dtype=np.int32)
info("source_id data indexes start from 1 not 0 !!!")

# source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))

s_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1
# plot3d(source_mask.data, model)

info("Number of unique affected points is: %d", len(nzinds[0])+1)

# Assert that first and last index are as expected
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1], 
       nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(s_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(f.data[0])))

info("-At this point source_mask and source_id have been popoulated correctly-")

nnz_shape = (grid.shape[0], grid.shape[1])  # Change only 3rd dim

nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x, y), space_order=0, dtype=np.int32)

nnz_sp_source_mask.data[:, :] = s_mask.data[:, :, :].sum(2)
inds = np.where(s_mask.data == 1.)
print("Grid - source positions:", inds)
maxz = len(np.unique(inds[-1]))
# Change only 3rd dim
sparse_shape = (grid.shape[0], grid.shape[1], maxz)

assert(len(nnz_sp_source_mask.dimensions) == (len(s_mask.dimensions)-1))

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

id_dim = Dimension(name='id_dim')
b_dim = Dimension(name='b_dim')

save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], source_id], expr=src * dt**2 / m)

op1 = Operator([save_src_term])

op1.apply(time=time_range.num-1)

print("========passs===========")

usol = TimeFunction(name="usol", grid=grid, space_order=so, time_order=2)
sp_zi = Dimension(name='sp_zi')

# import pdb; pdb.set_trace()

sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)),
                          dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

# Now holds IDs
sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])

assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 3)

t = grid.stepping_dim

zind = Scalar(name='zind', dtype=np.int32)
xb_size = Scalar(name='xb_size', dtype=np.int32)
yb_size = Scalar(name='yb_size', dtype=np.int32)
x0_blk0_size = Scalar(name='x0_blk0_size', dtype=np.int32)
y0_blk0_size = Scalar(name='y0_blk0_size', dtype=np.int32)

eq0 = Eq(sp_zi.symbolic_max, nnz_sp_source_mask[x, y] - 1, implicit_dims=(time, x, y))
eq1 = Eq(zind, sp_source_mask[x, y, sp_zi], implicit_dims=(time, x, y, sp_zi))

myexpr = s_mask[x, y, zind] * save_src[time, source_id[x, y, zind]]

eq2 = Inc(usol.forward[t+1, x, y, zind], myexpr, implicit_dims=(time, x, y, sp_zi))

import pdb;pdb.set_trace()
pde_2 = m * usol.dt2 - usol.laplace + model.damp * usol.dt
stencil_2 = Eq(usol.forward, solve(pde_2, usol.forward))

block_sizes = Function(name='block_sizes', shape=(4, ), dimensions=(b_dim,), space_order=0, dtype=np.int32)

# import pdb; pdb.set_trace()
block_sizes.data[:] = args.bsizes

# import pdb; pdb.set_trace()

performance_map = np.array([[0, 0, 0, 0, 0]])


bxstart = 8
bxend = 33
bystart = 8
byend = 33
bstep = 32


txstart = 16
txend = 33
tystart = 16
tyend = 33

tstep = 8


# Temporal autotuning
for tx in range(txstart, txend, tstep):
    # import pdb; pdb.set_trace()
    for ty in range(tystart, tyend, tstep):
        for bx in range(bxstart, bxend, bstep):
            for by in range(bystart, byend, bstep):

                block_sizes.data[:] = [tx, ty, bx, by]

                eqxb = Eq(xb_size, block_sizes[0])
                eqyb = Eq(yb_size, block_sizes[1])
                eqxb2 = Eq(x0_blk0_size, block_sizes[2])
                eqyb2 = Eq(y0_blk0_size, block_sizes[3])

                # import pdb; pdb.set_trace()
                # plot3d(source_mask.data, model)
                usol.data[:] = 0
                print("-----")
                op2 = Operator([eqxb, eqyb, eqxb2, eqyb2, stencil_2, eq0, eq1, eq2], opt=('advanced'))
                # print(op2.ccode)
                print("===Temporal blocking======================================")
                summary = op2.apply(time=time_range.num-1, dt=model.critical_dt)
                print("===========")

                performance_map = np.append(performance_map, [[tx, ty, bx, by, summary.globals['fdlike'].gflopss]], 0)

                normusol = norm(usol)
                print("===========")
                print(normusol)
                print("===========")
                # import pdb; pdb.set_trace()
                print("Norm(usol):", normusol)

print(performance_map)

tids = np.unique(performance_map[:, 0])

for tid in tids:
    bids = np.where((performance_map[:, 0] == tid) & (performance_map[:, 1] == tid))
    bx_data = np.unique(performance_map[bids, 2])
    by_data = np.unique(performance_map[bids, 3])
    gptss_data = performance_map[bids, 4]
    gptss_data = gptss_data.reshape(len(bx_data), len(by_data))

    fig, ax = plt.subplots()
    im = ax.imshow(gptss_data); pause(2)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(bx_data)))
    ax.set_yticks(np.arange(len(by_data)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(bx_data)
    ax.set_yticklabels(by_data)

    ax.set_title("Gpts/s for fixed tile size. (Sweeping block sizes)")
    fig.tight_layout()


# Loop over data dimensions and create text annotations.
#for i in range(len(bx_data)):
#    for j in range(len(by_data)):
#        text = ax.text(j, i, "{:.2f}".format(gptss_data[i, j]),
#                       ha="center", va="center", color="w")

# import pdb; pdb.set_trace()
    fig.colorbar(im, ax=ax)
    # ax = sns.heatmap(gptss_data, linewidth=0.5)
    # plt.savefig(str(shape[0]) + str(np.int32(tx)) + str(np.int32(ty) + ".pdf")



# save_src.data[0, source_id.data[14, 14, 11]]
# save_src.data[0 ,source_id.data[14, 14, sp_source_mask.data[14, 14, 0]]]

#plt.imshow(uref.data[2, int(nx/2) ,:, :]); pause(1)
