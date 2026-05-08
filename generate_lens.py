import numpy as np

from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.calculations import get_lens_distribution, circle_aperture, lens
from matplotlib import pyplot as plt


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    params.matrix_size = 1024
    params.pixel_size = 0.2
    params.distance = 268
    params.focal_length = params.distance
    frequency = 96.2  # GHz 96.2 GHz for 90 mW IMPATT, 95.3 for 900 mW IMPATT, 98 GHz for photomixing
    params.wavelength = params.get_wavelength_from_frequency(frequency)
    radius = 187/2  # radius in mm; 175 for PAPS; 187 mm for THz link
    n = 1.513 # refractive index of the lens material - COC

    phase_shift = -0.02

    phase = np.mod(phase_shift + get_lens_distribution(params),2*np.pi) * circle_aperture(params, radius, 1)

    filename = "outs/THz-link/Lens_v" + str(frequency) + "GHz_f" + str(params.distance) + "mm_px" + str(params.pixel_size) + "mm_r" + str(radius) + "mm.bmp"
    plt.imsave(filename, phase, cmap='gray', vmin=0, vmax=2*np.pi)


    r = np.arange(-radius, radius, params.pixel_size/10)
    phase_lens = np.mod(phase_shift + lens(r, params.focal_length, params.wavelength), 2 * np.pi)
    height_profile = phase_lens / 2 / np.pi * params.wavelength / (n - 1)
    print("Height:", max(height_profile))

    fig, ax = plt.subplots()

    plt.plot(r, height_profile, c='black', linewidth=0.1)
    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(filename.replace('.bmp', '.svg'))

