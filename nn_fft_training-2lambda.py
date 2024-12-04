import time
from datetime import datetime

import numpy as np

from lightprop.calculations import H_off_axis, H_on_axis, get_gaussian_distribution, get_lens_distribution, get_flat_array
from lightprop.lightfield import LightField
from lightprop.optimization.nn import NN_FFTTrainer
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
import tensorflow as tf

from matplotlib import pyplot as plt

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 256
    params.pixel_size = 0.8
    params.wavelength = PropagationParams.get_wavelength_from_frequency(180)

    out_path = "outs/TwoLambda/"

    # Define target optical field and input amplitude
    # In this example a simple focusing from wider Gaussian beam to the thinner one
    x0 = 30
    x1 = 20
    y0 = 0
    target0 = get_gaussian_distribution(params, x0, y0)
    target1 = get_gaussian_distribution(params, x1, y0)

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")
    plotter = Plotter1(
        LightField.from_complex_array(target0, params.wavelength, params.pixel_size)
        )
    plotter.save_output_amplitude(out_path + "Target0_" + str_current_datetime + ".bmp")
    # plotter.save_output_amplitude("outs/Target.bmp")

    plotter = Plotter1(
        LightField.from_complex_array(target1, params.wavelength, params.pixel_size)
        )
    plotter.save_output_amplitude(out_path + "Target1_" + str_current_datetime + ".bmp")
    # plotter.save_output_amplitude("outs/Target.bmp")

    params.beam_diameter = 30
    amp = get_gaussian_distribution(params, 0, 0)
    phase = np.array(
        [
            [0 for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

    plotter = Plotter1(
        LightField.from_complex_array(amp, params.wavelength, params.pixel_size)
        )
    plotter.save_output_amplitude(out_path + "Input_" + str_current_datetime + ".bmp")
    # plotter.save_output_amplitude("outs/Input.bmp")

    # Build NNTrainer or NNMultiTrainer
    NN = NN_FFTTrainer()

    
    # Run NN optimization
    # In case of NNMultiTrainer provide kernel as 3rd argument.
    # Please try running different numbers of iterations (last parameter)
    # Check the difference in the output for different amounts of training


    freqs = range(170,191,20)

    kernels_number = len(freqs)
    
    DWL = PropagationParams.get_wavelength_from_frequency(180)

    kernels = [np.empty([params.matrix_size,params.matrix_size], dtype="complex64")]*kernels_number 

    wavelength_scaling=[np.empty(1)]*kernels_number

    for i in range(len(freqs)):
        params.wavelength = PropagationParams.get_wavelength_from_frequency(freqs[i])
        kernels[i]=np.array(
            [
                [
                    H_on_axis(
                        x / params.pixel_size / params.pixel_size / params.matrix_size,
                        y / params.pixel_size / params.pixel_size / params.matrix_size,
                        params.distance,
                        params.wavelength,
                    )
                    for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
                ]
                for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
        )

        
        kernels[i]=LightField.from_complex_array(kernels[i], params.wavelength, params.pixel_size)
        
        wavelength_scaling[i] = DWL / params.wavelength


    
    initial_weights = np.mod(get_lens_distribution(params, x0),2*np.pi)

    trained_model = NN.optimize(
        [LightField(amp, phase, params.wavelength, params.pixel_size)]*kernels_number,
        [LightField(target0, phase, params.wavelength, params.pixel_size),LightField(target1, phase, params.wavelength, params.pixel_size)],
        kernels,
        wavelength_scaling,
        initial_weights,
        iterations=30000
    )

    

    # Plot loss vs epochs
    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    NN.plot_loss(out_path + "Loss_curve_" + str_current_datetime)
    

    # Extract the optimized phase map from the trainable layer
    optimized_phase = np.array(trained_model.layers[3].get_weights()[0])

    
    plotter = Plotter1(
        LightField(amp, optimized_phase, params.wavelength, params.pixel_size)
    )
    plotter.save_output_phase(out_path + "Structure_" + str_current_datetime + ".bmp")
   

    # Prepare input field and kernel
    field = np.array([amp, phase])
    field = field.reshape(
        (
            1,
            2,
            params.matrix_size,
            params.matrix_size,
        ),
        order="F",
    )

    
    for i in range(len(kernels)):
        params.wavelength = PropagationParams.get_wavelength_from_frequency(freqs[i])
        kernel = np.array([kernels[i].get_re(), kernels[i].get_im()])

        kernel = kernel.reshape(
            (
                1,
                2,
                params.matrix_size,
                params.matrix_size,
            ),
            order="F",
        )

        # Evaluate model on the input field
        result = trained_model([field, kernel, wavelength_scaling[i]]).numpy()
        result = result[0, 0, :, :] * np.exp(1j * result[0, 1, :, :])


        # Plot the result
        plotter = Plotter1(
        LightField.from_complex_array(result, params.wavelength, params.pixel_size)
        )
        plotter.save_output_amplitude(out_path + "ResultNN_" + str(freqs[i]) + "GHz_" + str_current_datetime + ".bmp")
        # plotter.save_output_amplitude("outs/ResultNN.bmp")

        


