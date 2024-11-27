# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""


import numpy as np

import time
from datetime import datetime

from PIL import Image

from lightprop.calculations import gaussian, get_gaussian_distribution, get_lens_distribution, get_tilted_wavefront
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxton
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
from lightprop.propagation import methods as prop
from matplotlib import pyplot as plt

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 0.4
    params.matrix_size = 1024
    params.pixel_size = 0.1
 
    DWL = PropagationParams.get_wavelength_from_frequency(180)   
    params.wavelength = DWL

    
    # Define input amplitude

    # params.beam_diameter = 2
    # amp = get_gaussian_distribution(params)


    

    # Import phase map of the structure
    
    # image = Image.open("outs/Zach/structure.bmp")
    # phase = np.asarray(image)[:,:,0]
    # phase = phase/255
    # phase = phase*2
    # phase = phase*np.pi

    angle = 85
    phase = get_tilted_wavefront(params, angle)

                
   # Calculate atomic field
    # shifts = [-20,-10,0,10,20]
    # amp = np.array(
    #     [
    #         [0 for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )
    # for i in shifts:
    #     for j in shifts:
    #         gauss = np.array(
    #             [
    #                 [
    #                     gaussian(np.sqrt((x - i) ** 2 + (y - j)**2), params.beam_diameter)
    #                     for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #                 ]
    #                 for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #             ]
    #         )
    #         amp = amp + gauss

    # field = LightField(amp, phase, params.wavelength, params.pixel_size)
    # plotter = Plotter1(field)
    # plotter.save_output_amplitude("outs/BT/input_field.png")

    
    # Import field

    image = Image.open("outs/BT/input_field.png")
    amp = np.asarray(image)[:,:,0]

    # propagate field
    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    params.distance = 500

    field = LightField(amp, phase, params.wavelength, params.pixel_size)
           
    result = prop.FFTPropagation().angular_space(field)

    plotter = Plotter1(result)
    plotter.save_output_amplitude("outs/BT/result_angle_"+str(angle)+"degrees.bmp")
    