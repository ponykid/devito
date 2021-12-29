from devito import Grid, TimeFunction, Eq, Operator, \
    Function, ConditionalDimension, Le

# Define a physical size
shape = (101, 101, 101)
extent = (1000., 1000., 1000.)

grid = Grid(shape=shape, extent=extent)

x, y, z = grid.dimensions

p = TimeFunction(name='p', grid=grid)

ax_x = Function(name='ax_x', grid=grid)

ax_x.data[:, :, 50] = 1

second_update = Le(ax_x, 1)

# Conditional masks for update
use_2nd = ConditionalDimension(name='use_2nd', parent=z,
                               condition=second_update)

eq_p = Eq(p.forward, 1, implicit_dims=use_2nd)

op = Operator([eq_p])
