# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""


import numpy as np

import time
from datetime import datetime

from PIL import Image

from lightprop.calculations import get_gaussian_distribution, circle_aperture
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxton
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.propagation import methods as prop
from matplotlib import pyplot as plt

from alive_progress import alive_bar

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 1024
    params.pixel_size = 0.01
    params.wavelength = 1030 * 10**-6
    params.focal_length = 50
    params.distance = params.focal_length
 
    
    # Define input amplitude

    params.beam_diameter = 0.5
    amp = get_gaussian_distribution(params)
    


    # Import phase map of the structure
    
    name = "px_0.005mm_1024_wavelength_1030nm_f_50mm"
    
    image = Image.open("outs/Nanochisel/FZP_" + name + ".bmp")
    phase = np.asarray(image)[:,:,0]
    phase = phase/255
    phase = phase*2
    phase = phase*np.pi

                
    # propagate field

    
    distance = 2 * params.distance
    distances = np.arange(1, distance, 1)
    kernels_number = len(distances)

    scale = round(distance / params.matrix_size / params.pixel_size)

    cross_section = np.empty([kernels_number * scale, params.matrix_size])
    
    obstacle = circle_aperture(params, 5, 0)

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    with alive_bar(kernels_number) as bar:
        for i in range(kernels_number):

            params.distance = distances[i]

            phase_loop = phase.copy()
            
            field = LightField(amp, phase_loop, params.wavelength, params.pixel_size)
            
            result = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)

            for j in range(scale):
                cross_section[i * scale + j] = result.amp[np.round(params.matrix_size/2).astype(int)]
            
            bar()


    # putting an obstacle

    # for i in range(kernels_number):

    #         if distances[i] <= 200:
    #             params.distance = distances[i]

    #             phase_loop = phase.copy()
            
    #             field = LightField(amp, phase_loop, params.wavelength, params.pixel_size)
            
    #             result = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)

    #         else:
    #             params.distance = 20
    #             phase_loop = phase.copy()        
    #             field = LightField(amp, phase_loop, params.wavelength, params.pixel_size)        
    #             field = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)

    #             params.distance = distances[i] - 20
    #             phase_loop = phase.copy()        
    #             field.amp = field.amp * obstacle
    #             field.phase = field.phase * obstacle     
    #             result = prop.FFTPropagation().propagate(field, params.distance, params.wavelength)


    #         cross_section[i] = result.amp[np.round(params.matrix_size/2).astype(int)]

    plt.imsave("outs/Nanochisel/xz_scan_" + name + ".bmp", cross_section, cmap='gray')
    