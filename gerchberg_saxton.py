# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""

from dis import dis
import numpy as np

from lightprop.calculations import gaussian, get_gaussian_distribution, get_lens_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxtonFFT
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter1, PlotTypes

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 512
    params.pixel_size = 0.4
    params.wavelength = params.get_wavelength_from_frequency(100)
    params.focal_length = 40
    params.distance = 60

    # Define target optical field and input amplitude
    # In this example two focal points placed outside the main optical axis
    # x_shift1 = 20
    # x_shift2 = 50
    # target = np.array(
    #     [
    #         [
    #             gaussian(np.sqrt((x - x_shift1) ** 2 + y**2), params.beam_diameter)
    #             for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #         ]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )
    # ) + np.array(
    #     [
    #         [
    #             gaussian(np.sqrt((x - x_shift2) ** 2 + y**2), params.beam_diameter)
    #             for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #         ]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )


    # Donut shape
    r_shift = 15
    target = np.array(
        [
            [
                gaussian(np.abs(np.sqrt(x**2 + y**2) - r_shift), 2*params.beam_diameter)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )
    # ) + np.array(
    #     [
    #         [
    #             gaussian(np.sqrt((x - x_shift2) ** 2 + y**2), params.beam_diameter)
    #             for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #         ]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )


    
    params.beam_diameter = 6.5
    amp = get_gaussian_distribution(params)
    phase = np.array(
        [
            [-1 for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

    # Prepare optimizer
    GS = GerchbergSaxtonFFT(params.distance)

    # Run optimizer
    input_plane, output_plane = GS.optimize(LightField(amp, phase, params.wavelength, params.pixel_size),
                                            LightField(target, phase, params.wavelength, params.pixel_size), iterations = 10)

    # Plot the result - optimized phase map
    plotter = Plotter1(input_plane)
    plotter.save_output_phase("outs/Zach/structure.bmp")

    # Plot the result - optimized phase map with focusing lens for divergent wave
    lens = get_lens_distribution(params)
    input_plane.phase = input_plane.phase + lens
    
    plotter = Plotter1(input_plane)
    plotter.save_output_phase("outs/Zach/structure_with_lens.bmp")


    # Plot the input amplitude
    plotter = Plotter1(input_plane)
    plotter.save_output_amplitude("outs/Zach/input_field.png")

    # Plot the result - output amplitude
    plotter = Plotter1(output_plane)

    plotter.save_output_amplitude("outs/Zach/result.png")

    # Plot the result - output phase
    plotter = Plotter1(output_plane)
    plotter.save_output_phase("outs/Zach/result_phase.png")

    # Plot the target amplitude
    plotter = Plotter1(LightField(target, phase, params.wavelength, params.pixel_size))
    plotter.save_output_amplitude("outs/Zach/target.png")
    plotter.show()
