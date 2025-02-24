from devito import *


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium through a 
    wave equation in terms of the P-wave velocity and the vector reflectivity.

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
    m = model.m
    vp = model.vp
    r = model.r

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    aco = m*u.dt2 - u.laplace + model.damp * u.dt
    pde = aco - 1/vp * grad(vp).T * grad(u) + 2 * r.T * grad(u)
    stencil = Eq(u.forward, solve(pde, u.forward))

    s = model.grid.stepping_dim.spacing

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2  / m)
    
    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    return Operator([stencil] + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


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
    vp = model.vp
    r = model.r

    v = TimeFunction(name='v', grid=model.grid, 
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing

    aco = m*v.dt2 - v.laplace + model.damp * v.dt.T
    eqn = m*v.dt2 - (v.laplace - div(grad(vp, .5) * v/vp, -.5) + div(2*r*v, -.5)) + model.damp * v.dt.T
    # eqn = aco + div(grad(vp, -.5) * v/vp, .5)
    # eqn = aco + div(grad(1/vp, -.5) * v*vp, .5)
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


# def GradientOperator(model, geometry, space_order=4, save=True,
#                      kernel='OT2', **kwargs):
#     """
#     Construct a gradient operator in an acoustic media.

#     Parameters
#     ----------
#     model : Model
#         Object containing the physical parameters.
#     geometry : AcquisitionGeometry
#         Geometry object that contains the source (SparseTimeFunction) and
#         receivers (SparseTimeFunction) and their position.
#     space_order : int, optional
#         Space discretization order.
#     save : int or Buffer, optional
#         Option to store the entire (unrolled) wavefield.
#     kernel : str, optional
#         Type of discretization, centered or shifted.
#     """
#     m = model.m

#     # Gradient symbol and wavefield symbols
#     grad = Function(name='grad', grid=model.grid)
#     u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
#                      else None, time_order=2, space_order=space_order)
#     v = TimeFunction(name='v', grid=model.grid, save=None,
#                      time_order=2, space_order=space_order)
#     rec = geometry.rec

#     s = model.grid.stepping_dim.spacing
#     eqn = iso_stencil(v, model, kernel, forward=False)

#     if kernel == 'OT2':
#         gradient_update = Inc(grad, - u * v.dt2)
#     elif kernel == 'OT4':
#         gradient_update = Inc(grad, - u * v.dt2 - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
#     # Add expression for receiver injection
#     receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

#     # Substitute spacing terms to reduce flops
#     return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
#                     name='Gradient', **kwargs)


# def BornOperator(model, geometry, space_order=4,
#                  kernel='OT2', **kwargs):
#     """
#     Construct an Linearized Born operator in an acoustic media.

#     Parameters
#     ----------
#     model : Model
#         Object containing the physical parameters.
#     geometry : AcquisitionGeometry
#         Geometry object that contains the source (SparseTimeFunction) and
#         receivers (SparseTimeFunction) and their position.
#     space_order : int, optional
#         Space discretization order.
#     kernel : str, optional
#         Type of discretization, centered or shifted.
#     """
#     m = model.m

#     # Create source and receiver symbols
#     src = geometry.src
#     rec = geometry.rec

#     # Create wavefields and a dm field
#     u = TimeFunction(name="u", grid=model.grid, save=None,
#                      time_order=2, space_order=space_order)
#     U = TimeFunction(name="U", grid=model.grid, save=None,
#                      time_order=2, space_order=space_order)
#     dm = Function(name="dm", grid=model.grid, space_order=0)

#     s = model.grid.stepping_dim.spacing
#     eqn1 = iso_stencil(u, model, kernel)
#     eqn2 = iso_stencil(U, model, kernel, q=-dm*u.dt2)

#     # Add source term expression for u
#     source = src.inject(field=u.forward, expr=src * s**2 / m)

#     # Create receiver interpolation expression from U
#     receivers = rec.interpolate(expr=U)

#     # Substitute spacing terms to reduce flops
#     return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
#                     name='Born', **kwargs)
