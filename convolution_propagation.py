# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""


import numpy as np

import time
from datetime import datetime

from PIL import Image

from lightprop.calculations import gaussian, get_gaussian_distribution, get_lens_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxton
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.propagation import methods as prop
from matplotlib import pyplot as plt

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 1024
    params.pixel_size = 0.2


    params.wavelength = PropagationParams.get_wavelength_from_nu(180)

    
    # Define input amplitude

    params.beam_diameter = 15
    amp = get_gaussian_distribution(params)
    

    # Import phase map of the structure
    
    # image = Image.open("outs/Structure_23.07.2024-10_45_09.bmp")
    
    # # convert image to numpy array
    # phase = np.asarray(image)[:,:,0]

    # phase=phase/255
    # phase=phase*2
    # phase=phase*np.pi

    params.focal_length = 120
    params.distance = params.focal_length
    phase = np.mod(get_lens_distribution(params),2*np.pi)

    
    # plt.imshow(amp, interpolation="nearest")
    # plt.show()
    
    # plt.imshow(phase, interpolation="nearest")
    # plt.show()
        
    # propagate field

    # kernel = prop.FFTPropagation().calculate_kernel(
    #         params.distance, params.wavelength, params.matrix_size, params.pixel_size
    #     )
    
    # plt.imshow(np.angle(kernel), interpolation="nearest")
    # plt.show()


    field = LightField(amp, phase, params.wavelength, params.pixel_size)


    result = prop.FFTPropagation().propagate(field, params.distance)

    # show results
    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")


    plotter = Plotter1(field)
    plotter.save_output_amplitude("outs/Input_" + str_current_datetime + ".bmp")

    plotter = Plotter1(result)
    plotter.save_output_amplitude("outs/Result_" + str_current_datetime + ".bmp")