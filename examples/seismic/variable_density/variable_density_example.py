import numpy as np
try:
    import pytest
except ImportError:
    pass

from devito.logger import info
from devito import Constant, Function, smooth, norm
from examples.seismic.variable_density import VariableDensityAcousticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args, Receiver


def variable_density_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., kernel='OT2', space_order=4, nbl=10,
                   preset='layers-isotropic', fs=False, **kwargs):
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing,
                       fs=fs, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn, **kwargs)

    # Create solver object to provide relevant operators
    solver = VariableDensityAcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver