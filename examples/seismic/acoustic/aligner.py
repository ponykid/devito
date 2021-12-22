import numpy as np
from matplotlib.pyplot import pause # noqa
from devito import (Dimension, Function, TimeFunction, Eq, Inc,
                    Operator, norm)  # noqa
from devito.types import Scalar
from devito.tools import as_list
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

__all__ = ['aligner', 'term_aligner']


def aligner(grid, v, src, dt, time_range):

    x, y = grid.dimensions
    time = grid.time_dim
    t = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, space_order=2)
    m = Function(name='m', grid=grid)
    m.data[:] = 1./(v*v)

    # Injection to field u.forward
    src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
    op = Operator(src_term)
    op(time=time_range.num-1, dt=dt)
    norm_ref = norm(u)
    print(norm_ref)
    # u2 = u

    # Get the nonzero indices to nzinds tuple
    nzinds = np.nonzero(u.data[0])
    assert len(nzinds) == len(grid.shape)

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
    # zinds = np.column_stack(nzinds)

    sparse_shape = as_list((grid.shape[0], maxz))  # Change only 2nd dim

    sp_yi = Dimension(name='sp_yi')
    sp_sm = Function(name='sp_sm', shape=sparse_shape, dimensions=(x, sp_yi),
                     dtype=np.int32)

    # Now holds IDs
    sp_sm.data[nzinds[0], :] = tuple(nzinds[1][:len(np.unique(nzinds[1]))])

    # assert(np.count_nonzero(sp_sm.data) == len(nzinds[0]))
    assert(len(sp_sm.dimensions) == 2)

    yind = Scalar(name='yind', dtype=np.int32)

    eq0 = Eq(sp_yi.symbolic_max, nnz_mask[x] - 1, implicit_dims=(time, x))
    eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
    # eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
    myexpr = s_mask[x, yind] * save_src[time, source_id[x, yind]]
    eq2 = Inc(u.forward[t, x, yind], myexpr, implicit_dims=(time, x, sp_yi))

    src_term2 = [eq0, eq1, eq2]

    print("End of aligner", norm_ref)

    return src_term2, u


def term_aligner(u, src, src_term, geometry):

    grid = src_term.field.grid

    x, y, z = grid.dimensions
    time = grid.time_dim
    t = grid.stepping_dim

    u = TimeFunction(name="u", grid=grid, space_order=2)
    # m = Function(name='m', grid=grid)
    # m.data[:] = 1./(v*v)

    # Injection to field u.forward

    src_term2 = src.inject(field=u.forward, expr=src_term.expr)

    op = Operator(src_term2)
    time_range = geometry.time_axis
    op(time=time_range.num-1, dt=geometry.dt)
    # norm_ref = norm(u)
    # print(norm_ref)
    # u2 = u
    import pdb;pdb.set_trace()
    # Get the nonzero indices to nzinds tuple
    nzinds = np.nonzero(u.data[0])
    assert len(nzinds) == len(grid.shape)

    import pdb;pdb.set_trace()

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
    # zinds = np.column_stack(nzinds)

    sparse_shape = as_list((grid.shape[0], maxz))  # Change only 2nd dim

    sp_yi = Dimension(name='sp_yi')
    sp_sm = Function(name='sp_sm', shape=sparse_shape, dimensions=(x, sp_yi),
                     dtype=np.int32)

    # Now holds IDs
    sp_sm.data[nzinds[0], :] = tuple(nzinds[1][:len(np.unique(nzinds[1]))])

    # assert(np.count_nonzero(sp_sm.data) == len(nzinds[0]))
    assert(len(sp_sm.dimensions) == 2)

    yind = Scalar(name='yind', dtype=np.int32)

    eq0 = Eq(sp_yi.symbolic_max, nnz_mask[x] - 1, implicit_dims=(time, x))
    eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
    # eq1 = Eq(yind, sp_sm[x, sp_yi], implicit_dims=(time, x, sp_yi))
    myexpr = s_mask[x, yind] * save_src[time, source_id[x, yind]]
    eq2 = Inc(u.forward[t, x, yind], myexpr, implicit_dims=(time, x, sp_yi))

    src_term2 = [eq0, eq1, eq2]

    print("End of aligner", norm_ref)

    return src_term2, u
