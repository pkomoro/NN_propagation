import numpy as np

from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.calculations import get_lens_distribution, circle_aperture, lens
from matplotlib import pyplot as plt


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    params.matrix_size = 2048
    params.pixel_size = 0.1
    params.distance = 500
    params.focal_length = params.distance
    frequency = 250  # GHz
    params.wavelength = params.get_wavelength_from_frequency(frequency)  # 98 GHz
    radius = 175/2  # radius in mm; 175 for PAPS; 187 mm for THz link
    n = 1.52 # refractive index of the lens material - COC

    phase = np.mod(get_lens_distribution(params),2*np.pi) * circle_aperture(params, radius, 1)

    filename = "outs/FENG-PAPS/Lens_v" + str(frequency) + "GHz_f" + str(params.distance) + "mm_px" + str(params.pixel_size) + "mm_r" + str(radius) + "mm.bmp"
    plt.imsave(filename, phase, cmap='gray', vmin=0, vmax=2*np.pi)


    r = np.arange(-radius, radius, params.pixel_size)
    phase_lens = np.mod(lens(r, params.focal_length, params.wavelength), 2 * np.pi)
    height_profile = phase_lens / 2 / np.pi * params.wavelength / (n - 1)
    print("Height:", max(height_profile))
    plt.plot(r, height_profile, label='Lens height profile')
    plt.axis('off')
    plt.savefig(filename.replace('.bmp', '.svg'))

