# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""


import numpy as np

import time
from datetime import datetime

from PIL import Image

from lightprop.calculations import gaussian, get_gaussian_distribution, get_lens_distribution, get_FZP_distribution, get_FZP_radii
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxton
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.propagation import methods as prop
from matplotlib import pyplot as plt

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 30
    params.matrix_size = 256
    params.pixel_size = 0.9
    freq = 300
    params.wavelength = params.get_wavelength_from_frequency(freq)
    params.focal_length = 200
    params.distance = params.focal_length
 
        
    # Define input amplitude

    amp = get_gaussian_distribution(params)

    phase = get_lens_distribution(params)
    
    

    field = LightField(amp, phase, params.wavelength, params.pixel_size)

    # current_datetime = datetime.now()
    # str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    name = "px_" + str(params.pixel_size) + "mm_size_" + str(params.matrix_size) + "_frequency" + str(freq) + "GHz_f_" + str(params.focal_length) + "mm"

    plotter = Plotter1(field)
    plotter.save_output_phase("outs/simple_lens/lens_" + name + ".bmp")
    plotter.save_output_intensity("outs/simple_lens/Input_" + name + ".bmp")

    # Import phase map of the structure
    
    # image = Image.open("outs/Zach/structure.bmp")
    # phase = np.asarray(image)[:,:,0]
    # phase = phase/255
    # phase = phase*2
    # phase = phase*np.pi


    # propagate field
              
    result = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)

    plotter = Plotter1(result)
    plotter.save_output_intensity("outs/simple_lens/Focal_plane_" + name + ".bmp")

    
    