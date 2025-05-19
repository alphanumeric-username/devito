from devito import *
from examples.seismic import Model, setup_geometry

from examples.seismic.preset_models import Gardners
from examples.seismic.vector_reflectivity.model_utils import load_model_from_yaml, extract_model_args

__all__ = ['model_layer', 'model_marmousi', 'model_marmousi2']

def layer_cake_vp(vp_list: list, shape, dtype = np.float64):
    vp = np.ones(shape, dtype)
    nlayers = len(vp_list)
    
    for i, vpi in enumerate(vp_list):
        idx_top = i*shape[1]//nlayers
        idx_bottom = (i+1)*shape[1]//nlayers
        vp[:, idx_top : idx_bottom] *= vpi
    
    return vp


def model_layer(vp_list, dtype=np.float64):
    # Parameters
    NX = 251
    NZ = 251

    SHAPE = (NX, NZ)
    ORIGIN = (0, 0)
    SPACING = (4, 4)
    SPACE_ORDER = 2
    NBL = 50
    # vp_list = [1.5, 2.0, 3.5, 4.0, 3.0, 6.0, 7.0, 7.5, 8.0]
    vp = layer_cake_vp(vp_list, SHAPE, dtype)

    b = Gardners(vp, normalize=True)
    model = Model(
        vp=vp, b=b, origin=ORIGIN, spacing=SPACING, shape=SHAPE, space_order=SPACE_ORDER, 
        nbl=NBL, bcs='damp', dtype=dtype, staggered=NODE
    )

    return model


def model_marmousi(path, dtype=np.float64):
    # model = load_model_from_yaml('/home/filipe/projects/models/marmousi-resample/header.yaml', dtype=np.float64)
    model = load_model_from_yaml(path, dtype=dtype)
    model_args = extract_model_args(model)
    model_args['b'] = Gardners(model_args.get('vp'))
    # model_args['b'] = Gardners(model_args.get('vp'), False)
    model = Model(staggered=NODE, **model_args)

    return model

