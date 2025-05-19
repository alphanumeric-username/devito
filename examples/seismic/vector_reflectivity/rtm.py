from devito import *
from examples.seismic.model import Model
from examples.seismic.source import PointSource, Receiver, RickerSource
from examples.seismic.utils import setup_geometry
from examples.seismic.vector_reflectivity.model_creators import model_layer, model_marmousi

import os, shutil, gc

TIME_ORDER=2

def main():
    # model = model_layer([1.5, 2.5, 3.0, 4.0, 5.0], np.float32)
    marmopath = '/home/filipe/projects/models/marmousi-resample/header.yaml'
    model = model_marmousi(marmopath, np.float32)
    # model.vp.data[model.nbl:-model.nbl,model.nbl:-model.nbl].tofile('temp.bin')
    f0 = 0.030
    tn = 4000

    geometry = setup_geometry(model, tn, f0)
    geometry.resample(model.critical_dt)
    print(geometry.nt)
    # return

    watermask = 1 - ((model.vp.data <= 1.5).astype(np.float32))

    # imf = img_final(model, geometry, nshots=10, src_depth=2*model.spacing[1], rec_depth=model.spacing[1])
    imf = img_final2(model, geometry, nshots=10, rec_depth=model.spacing[1])
    imf_lap = lapla(model, imf)
    imf_outfile = f'image_nx{model.shape[0]}_nz{model.shape[1]}.bin'
    lap_outfile = f'lap_nx{model.shape[0]}_nz{model.shape[1]}.bin'
    (imf.data * watermask)[model.nbl:-model.nbl, model.nbl:-model.nbl].tofile(imf_outfile)
    (imf_lap.data * watermask)[model.nbl:-model.nbl, model.nbl:-model.nbl].tofile(lap_outfile)


# Function to create sources and receivers
def src_rec(model, geometry, src_pos, rec_depth, nrec):
    src = RickerSource(name='src', grid=model.grid, f0=geometry.f0, 
                       time_range=geometry.time_axis, npoint=1)

    rec = Receiver(name='rec', grid=model.grid, npoint=nrec,
                   time_range=geometry.time_axis)

    src.coordinates.data[:, 0] = src_pos[0]
    src.coordinates.data[:, 1] = src_pos[1]
    rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1] = rec_depth
    
    return src, rec


# Function to create adjoint sources and receivers
def adjoint_src(model, geometry, src_pos):
    srca = PointSource(name='srca', grid=model.grid,
                       time_range=geometry.time_axis,
                       npoint=1)

    srca.coordinates.data[:, 0] = src_pos[0]
    srca.coordinates.data[:, 1] = src_pos[1]

    return srca


def forward(model, geometry, src, rec):
    u = TimeFunction(name='u', grid=model.grid, time_order=TIME_ORDER,
                            space_order=model.space_order, staggered=NODE, save=geometry.nt)

    m = model.m
    r = model.r
    vp = model.vp
    dt = model.grid.stepping_dim.spacing

    pde = m * u.dt2 - u.laplace + (- 1/vp*grad(vp) + 2*r).T * grad(u) + model.damp * u.dt

    stencil = [Eq(u.forward, solve(pde, u.forward))]

    src_term = src.inject(field=u.forward, expr=src * dt**2/model.m)
    rec_term = rec.interpolate(expr=u)

    op = Operator(stencil + src_term + rec_term, subs=model.spacing_map)
    op(dt=geometry.dt)
    
    return u


def adjoint(model: Model, geometry, rec, srca=None):
    v = TimeFunction(name='v', grid=model.grid, time_order=TIME_ORDER,
                    space_order=model.space_order, staggered=NODE, save=geometry.nt)
    m = model.m
    vp = model.vp
    r = model.r
    dt = model.grid.stepping_dim.spacing

    pde = m * v.dt2 - v.laplace - div((-grad(vp)/vp + 2*r) * v, .5) + model.damp * v.dt.T
    
    stencil = [Eq(v.backward, solve(pde, v.backward))]

    rec_term = rec.inject(v.backward, expr = rec * dt**2 / m)

    terms = stencil
    terms += rec_term

    if srca:
        srca_term = srca.interpolate(expr=v)
        terms += srca_term

    op = Operator(terms, subs=model.spacing_map)
    op(dt=geometry.dt)
    return v


def imaging_condition(u: np.ndarray, v: np.ndarray, cumulative = False) -> np.array:
    """
    Implements the Claerbout's imaging condition.

    Parameters
    ----------
    u : np.ndarray
        The forward wavefield.
    v : np.ndarray
        The adjoint wavefield.
    cumulative : bool
        If `False`, only the final image will be returned. If `True`, returns the cumulative sum of the imaging condition along each time step.
    """
    H = np.sum(u**2, axis=0)

    if cumulative:
        return np.cumsum(u * v, axis=0) / (H + 1e-9)
    else:
        return np.sum(u * v, axis=0) / (H + 1e-9)


def imaging_condition2(wavedir: str, shape, nt: int, dtype) -> np.array:
    H = np.zeros(shape)
    I = np.zeros(shape)

    for i in range(nt):
        ui = np.fromfile(f'{wavedir}/u{i}.bin', dtype=dtype).reshape(shape)
        vi = np.fromfile(f'{wavedir}/v{i}.bin', dtype=dtype).reshape(shape)
        H += ui*ui
        I += ui*vi

    return I/(H + 1e-9)


def img_final(model, geometry, src_depth=10., nshots=10, shot_offset=0., rec_depth=10., cumulative = False, wavedir='wave'):
    source_locations = np.empty((nshots, 2), dtype=np.float32)
    # source_locations[:, 0] = np.linspace(0., model.shape[0]*model.spacing[0],
    source_locations[:, 0] = np.linspace(0., model.domain_size[0],
                                        num=nshots) + shot_offset
    source_locations[:, 1] = src_depth

    if cumulative:
        imf = TimeFunction(name="imf", grid=model.grid, space_order=model.space_order, time_order=TIME_ORDER, save=geometry.nt)
    else:
        imf = Function(name="imf", grid=model.grid, space_order=model.space_order)

    if os.path.exists(wavedir):
        shutil.rmtree(wavedir)

    os.mkdir(wavedir)

    for i in range(nshots):
        print('Imaging source %d out of %d' % (i+1, nshots))
        src_pos = source_locations[i, :]
        
        src, rec = src_rec(model, geometry, src_pos, rec_depth, model.shape[0])
        
        u = forward(model, geometry, src, rec)
        nt = u.shape[0]
        for i in range(u.shape[0]):
            u.data[i].tofile(f'{wavedir}/u{i}.bin')
        
        del u
        gc.collect()
        
        # u.data.tofile('u.temp')
        v = adjoint(model, geometry, rec)
        for i in range(v.shape[0]):
            v.data[i].tofile(f'{wavedir}/v{i}.bin')
        
        del v
        gc.collect()

        # u.data.tofile('u.temp')
        # imf.data[:] += imaging_condition(u.data, v.data, cumulative)
        shape_ = (model.shape[0] + 2*model.nbl, model.shape[0] + 2*model.nbl)
        imf.data[:] += imaging_condition2(wavedir, shape_, nt, model.dtype)

    return imf


def img_final2(model, geometry, nshots=10, rec_depth=10):
    source_locations = np.empty((nshots, 2), dtype=np.float32)
    source_locations[:, 0] = np.linspace(0., model.shape[0]*model.spacing[0],
                                        num=nshots)
    source_locations[:, 1] = 10.

    imf = Function(name="imf", grid=model.grid, space_order=model.space_order)

    for i in range(nshots):
        print('Imaging source %d out of %d' % (i+1, nshots))
        pos = source_locations[i, :]
        src, rec = src_rec(model, geometry, pos, rec_depth, model.shape[0])
        u = forward(model, geometry, src, rec)

        op_imaging = ImagingOperator(model, geometry, imf, u)
        op_imaging(dt=model.critical_dt, rec=rec)
        del u
        gc.collect

    return imf


def ImagingOperator(model, geometry, imf, u, **kwargs):

    v = TimeFunction(name='v', grid=model.grid, space_order=model.space_order,
                           time_order=TIME_ORDER)

    m = model.m
    vp = model.vp
    r = model.r
    dt = model.grid.stepping_dim.spacing

    pde = m * v.dt2 - v.laplace - div((-grad(vp)/vp + 2*r) * v, .5) + model.damp * v.dt.T
    stencil = [Eq(v.backward, solve(pde, v.backward))]

    b = model.b
    # Define residual injection at the location of the forward receivers
    rec = PointSource(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)

    H = Function(name="H", grid=model.grid)

    H_sum = [Eq(H, H + u**2)]

    rec_term = rec.inject(field=u.backward, expr=rec * dt**2 / m)

    imf_update = [Eq(imf, imf + (v * u)/(H + 10**(-9)))]

    return Operator(stencil + rec_term + H_sum + imf_update, subs=model.spacing_map,
                    name='Imaging')



def lapla(model, image):
    lapla = Function(name='lapla', grid=model.grid, space_order=model.space_order)
    stencil = Eq(lapla, -image.laplace)
    op = Operator([stencil])
    op.apply()
    return lapla
    


if __name__ == '__main__':
    main()