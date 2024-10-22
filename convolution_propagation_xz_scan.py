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
    params.matrix_size = 256
    params.pixel_size = 0.8
 
    
    # Define input amplitude

    params.beam_diameter = 25
    amp = get_gaussian_distribution(params)
    

    # Import phase map of the structure
    
    image = Image.open("outs/Zach/structure.bmp")
    phase = np.asarray(image)[:,:,0]
    phase = phase/255
    phase = phase*2
    phase = phase*np.pi

                
    # propagate field

    
    distances = range(20,400,1)
    kernels_number = len(distances)

    cross_section = np.empty([kernels_number, params.matrix_size])
    

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    for i in range(kernels_number):

        params.distance = distances[i]

        phase_loop = phase.copy()
        
        field = LightField(amp, phase_loop, params.wavelength, params.pixel_size)
        
        result = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)
        cross_section[i] = result.amp[np.round(params.matrix_size/2).astype(int)]


    plt.imsave("outs/Zach/xz_scan.bmp", cross_section, cmap='gray')
    