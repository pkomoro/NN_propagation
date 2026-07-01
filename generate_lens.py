import numpy as np

from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.calculations import get_lens_distribution, circle_aperture, lens, Fresnel_reflection
from matplotlib import pyplot as plt


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    params.matrix_size = 1024
    params.pixel_size = 0.2
    params.distance = 118
    params.focal_length = params.distance
    frequency = 98  # GHz 96.2 GHz for 90 mW IMPATT, 95.3 for 900 mW IMPATT, 98 GHz for photomixing
    params.wavelength = params.get_wavelength_from_frequency(frequency)
    radius = 187/2  # radius in mm; 175 for PAPS; 187 mm for THz link
    n = 1.513 # refractive index of the lens material - COC

    # phase = np.mod(get_lens_distribution(params),2*np.pi) * circle_aperture(params, radius, 1)

    filename = "outs/THz-link/Lens_v" + str(frequency) + "GHz_f" + str(params.distance) + "mm_px" + str(params.pixel_size) + "mm_r" + str(radius) + "mm.bmp"
    # plt.imsave(filename, phase, cmap='gray', vmin=0, vmax=2*np.pi)


    r = np.arange(-radius, radius, params.pixel_size/10)
    phase_unwrapped = lens(r, params.focal_length, params.wavelength)
    phase_lens = np.mod(phase_unwrapped, 2 * np.pi)
    
    height_profile = phase_lens / 2 / np.pi * params.wavelength / (n - 1)
    print("Height:", max(height_profile))

    
    fig, ax = plt.subplots()
    plt.plot(r, height_profile, c='black', linewidth=0.1)
    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(filename.replace('.bmp', '.svg'))
    plt.close()

    
    r = np.arange(0, radius, params.pixel_size/10)
    phase_unwrapped = lens(r, params.focal_length, params.wavelength)
    height_profile_unwrapped = phase_unwrapped / (2 * np.pi) * params.wavelength / (n - 1)
    
    gradient = np.gradient(height_profile_unwrapped, r)
    angle_deg = np.degrees(-np.arctan(gradient))


    Rs = Fresnel_reflection(1, n, np.radians(angle_deg), True)
    Rp = Fresnel_reflection(1, n, np.radians(angle_deg), False) 

    # fig3, ax3 = plt.subplots()
    # ax3.plot(r, Rs, c='black', linewidth=1)
    # ax3.plot(r, Rp, c='green', linewidth=1)
    # ax3.set_title('Fresnel reflection vs angle')
    # ax3.set_xlabel('Radial position (mm)')
    # ax3.set_ylabel('Reflectance')
    # ax3.grid(True)
    # plt.show()

    # Create matrix in polar coordinates
    r_matrix = np.linspace(0, radius, 256)
    theta_matrix = np.linspace(0, 2*np.pi, 256)
    R_mesh, Theta_mesh = np.meshgrid(r_matrix, theta_matrix)
    
    # Calculate Rs for every point in the matrix
    angle_deg_matrix = np.interp(R_mesh, r, angle_deg)
    
    Rs_matrix = Fresnel_reflection(1, n, np.radians(angle_deg_matrix), True) * np.sin(Theta_mesh)**2 + Fresnel_reflection(1, n, np.radians(angle_deg_matrix), False) * np.cos(Theta_mesh)**2
    
    # Plot Rs matrix in polar coordinates
    fig4, ax4 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    im = ax4.contourf(Theta_mesh, R_mesh, Rs_matrix, levels=100, cmap='viridis')

    ax4.set_title('Fresnel Reflection')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Reflectance')
    plt.savefig(filename.replace('.bmp', '_FresnelReflection.jpg'), dpi=1000, bbox_inches='tight')
    plt.savefig(filename.replace('.bmp', '_FresnelReflection.svg'), bbox_inches='tight')








