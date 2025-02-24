from devito import *

from examples.seismic import AcquisitionGeometry


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium with variable density.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec


    # Create the stencil
    kappa = 1/(model.m * model.b)
    pde = u.dt2 - kappa * div(model.b * grad(u, shift=.5), shift=-.5) + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Create the equation
    src_term = src.inject(u.forward, expr = src*model.critical_dt**2/model.m)
    rec_term = rec.interpolate(expr=u.forward)

    equation = [stencil] + src_term + rec_term

    return Operator(equation, subs=model.spacing_map, name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', save=False, **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    v = TimeFunction(name='v', grid=model.grid, 
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing

    kappa = 1/(model.b * model.m)
    eqn = v.dt2 - div(model.b * grad(kappa * v, shift=0.5), shift=-0.5) + model.damp * v.dt.T
    stencil = Eq(v.backward, solve(eqn, v.backward))

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    # return Operator([stencil], subs=model.spacing_map,
    #                 name='Adjoint', **kwargs)
    return Operator([stencil] + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)

def BornOperator(model, geometry: AcquisitionGeometry, space_order=4, kernel='OT2', **kwargs):
    """
    Construct an Linearized Born operator in an acoustic medium with variable density.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    src = geometry.src
    rec = geometry.rec

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order = space_order, save=None)
    du = TimeFunction(name='du', grid=model.grid, time_order=2, space_order = space_order, save=None)
    dm = Function(name='dm', grid=model.grid, space_order=0)
    
    kappa = 1/(model.m * model.b)
    pde_bkg = u.dt2 - kappa*div(model.b * grad(u, shift=-.5), shift=.5) + model.damp * u.dt
    pde_sct = du.dt2 - kappa*div(model.b * grad(du, shift=-.5), shift=.5) + model.damp * du.dt \
        + dm/model.m * u.dt2

    eq_fwd = Eq(u.forward, solve(pde_bkg, u.forward))
    eq_born = Eq(du.forward, solve(pde_sct, du.forward))

    src_term = src.inject(u.forward, src*model.critical_dt**2/model.m )
    # src_term = src.inject(u.forward, src*model.critical_dt**2)
    rec_term = rec.interpolate(expr=du)

    return Operator([eq_fwd] + src_term + [eq_born] + rec_term, subs=model.spacing_map,
                    name='Born', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m
    b = model.b
    rec = geometry.rec

    gradient = Function(name = 'grad', grid = model.grid)
    u = TimeFunction(
        name = 'u', 
        grid=model.grid, 
        save= save and geometry.nt or None, 
        time_order=2, 
        space_order=space_order
    )

    v = TimeFunction(
        name = 'v', 
        grid=model.grid, 
        time_order=2, 
        space_order=space_order
    )

    kappa = 1/(m * b)
    pde = v.dt2 - div(b * grad(kappa * v, shift=.5), shift=-.5) + model.damp * v.dt.T

    eqn = Eq(v.backward, solve(pde, v.backward))

    gradient_update = Inc(gradient, -u.dt2 * v)  # grad += d²u/dt² * v

    rec_term = rec.inject(field = v.backward, expr = rec * model.critical_dt**2/m)  # Adjoint source term

    return Operator([eqn] + rec_term + [gradient_update], 
                    subs=model.spacing_map, name='Gradient', **kwargs)