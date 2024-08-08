import numpy as np

from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.calculations import get_lens_distribution, circle_aperture
from matplotlib import pyplot as plt


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    params.matrix_size = 256
    params.pixel_size = 0.8
    radius = 200

    phase = np.mod(get_lens_distribution(params),2*np.pi) * circle_aperture(params, radius)

    plt.imsave("outs/Lens_f" + str(params.distance) + "mm_px" + str(params.pixel_size) + "mm_r" + str(radius) + "mm.bmp", phase, cmap='gray', vmin=0, vmax=2*np.pi)
