from devito import *
from examples.seismic import Model

import numpy as np
import yaml

import json
import os


def clone_model(model: Model):
    kwargs = extract_model_args(model)
    return Model(**kwargs)


def extract_model_args(model: Model) -> dict:
    """
    Extract the arguments passed to the `Model` constructor from the corresponding object.
    """

    kwargs = {
        'vp': model.vp.data[model.nbl:-model.nbl, model.nbl:-model.nbl],
        'origin': model.origin, 
        'shape': model.shape, 
        'spacing': model.spacing, 
        'space_order': model.space_order,
        'grid': model.grid,
        'nbl': model.nbl,
        'bcs': 'damp',
        'dtype': model.dtype
    }

    if getattr(model, 'b', None) != None:
        kwargs['b'] = model.b.data[model.nbl:-model.nbl, model.nbl:-model.nbl]

    return kwargs


def load_model_from_json(header_filename: str, **kwargs) -> Model:
    with open(header_filename, 'r') as fin:
        header = json.load(fin)
        _load_from_header(header, header_filename,  **kwargs)


def load_model_from_yaml(header_filename: str, **kwargs) -> Model:
    with open(header_filename, 'r') as fin:
        header = yaml.load(fin, yaml.Loader)
        return _load_from_header(header, header_filename, **kwargs)
        
    
def _load_from_header(header: dict, header_filename: str, dtype=np.float32):
    nx = header.get('nx')
    nz = header.get('nz')
    
    dx = header.get('dx')
    dz = header.get('dz')

    origin = header.get('origin', (0., 0.))
    origin = (origin[0], origin[1])
    
    in_dtype = header.get('dtype', None)
    in_dtype = (in_dtype and np.dtype(in_dtype)) or np.float32

    space_order = header.get('space_order')
    nbl = header.get('nbl')

    vp = None

    if os.path.isabs(header.get('vp')):
        vp = np.fromfile(header.get('vp'), dtype=in_dtype)
    else:
        header_dir = os.path.dirname(header_filename)
        vp_filename = os.path.join(header_dir, header.get('vp'))

        vp = np.fromfile(vp_filename, dtype=in_dtype).reshape(nx, nz)
    
    vp = vp.astype(dtype)
    
    if header.get('vp_unit', 'km/s') == 'm/s':
        vp = vp/1000


    kwargs = {
        'vp': vp,
        'origin': origin,
        'shape': (nx, nz), 
        'spacing': (dx, dz), 
        'space_order': space_order,
        'nbl': nbl,
        'bcs': 'damp',
        'dtype': dtype
    }

    return Model(**kwargs)