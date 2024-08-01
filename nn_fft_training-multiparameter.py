import time
from datetime import datetime

import numpy as np

from lightprop.calculations import H_off_axis, H_on_axis, get_gaussian_distribution, get_lens_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.nn import NN_FFTTrainer
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes
import tensorflow as tf

from matplotlib import pyplot as plt

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 4
    params.matrix_size = 256
    params.pixel_size = 1.8
    params.wavelength = PropagationParams.get_wavelength_from_frequency(180)

    # Define target optical field and input amplitude
    # In this example a simple focusing from wider Gaussian beam to the thinner one
    x0 = 0
    y0 = 0
    target = get_gaussian_distribution(params, x0, y0)

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")
    plotter = Plotter1(
        LightField.from_complex_array(target, params.wavelength, params.pixel_size)
        )
    plotter.save_output_amplitude("outs/Target_" + str_current_datetime + ".bmp")
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
    plotter.save_output_amplitude("outs/Input_" + str_current_datetime + ".bmp")
    # plotter.save_output_amplitude("outs/Input.bmp")

    # Build NNTrainer or NNMultiTrainer
    NN = NN_FFTTrainer()

    
    # Run NN optimization
    # In case of NNMultiTrainer provide kernel as 3rd argument.
    # Please try running different numbers of iterations (last parameter)
    # Check the difference in the output for different amounts of training


    kernels_number=11
    wavelength_start=PropagationParams.get_wavelength_from_frequency(170)
    wavelength_stop=PropagationParams.get_wavelength_from_frequency(180)

    wavelength_step=(wavelength_stop-wavelength_start)/(kernels_number-1)
    kernels = [np.empty([params.matrix_size,params.matrix_size], dtype="complex64")]*kernels_number 
    # kernels_shifted = [np.empty([params.matrix_size,params.matrix_size], dtype="complex64")]*kernels_number


    for i in range(len(kernels)):
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
        # kernels_shifted[i] = np.array(np.roll(kernels[i], (int(params.matrix_size/2), int(params.matrix_size/2)), axis = (0,1)))

        
        kernels[i]=LightField.from_complex_array(kernels[i], params.wavelength, params.pixel_size)
        # kernels_shifted[i]=LightField.from_complex_array(kernels_shifted[i], params.wavelength, params.pixel_size)
        
        params.wavelength+=wavelength_step

    # figure, axis = plt.subplots(1, 2) 
    # axis[0].imshow(kernels[0].get_amplitude(), interpolation="nearest")
    # axis[1].imshow(kernels[0].get_phase(), interpolation="nearest")
    # plt.show()

    # figure, axis = plt.subplots(1, 2) 
    # axis[0].imshow(kernels_shifted[0].get_amplitude(), interpolation="nearest")
    # axis[1].imshow(kernels_shifted[0].get_phase(), interpolation="nearest")
    # plt.show()

    initial_weights = np.mod(get_lens_distribution(params),2*np.pi)

    trained_model = NN.optimize(
        [LightField(amp, phase, params.wavelength, params.pixel_size)]*kernels_number,
        [LightField(target, phase, params.wavelength, params.pixel_size)]*kernels_number,
        kernels,
        initial_weights,
        iterations=10000
    )

    

    # Plot loss vs epochs
    NN.plot_loss()

    
    # weights = trained_model.layers[3].get_weights()
    # weights[0]=get_lens_distribution(params) 
    # trained_model.layers[3].set_weights(weights)

    

    # Extract the optimized phase map from the trainable layer
    optimized_phase = np.array(trained_model.layers[3].get_weights()[0])

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    plotter = Plotter1(
        LightField(amp, optimized_phase, params.wavelength, params.pixel_size)
    )
    plotter.save_output_phase("outs/Structure_" + str_current_datetime + ".bmp")
    # plotter.save_output_phase("outs/Structure.bmp")

    # # Plot the target amplitude
    # plotter = Plotter(LightField(target, phase, params.wavelength, params.pixel_size), output_type=PlotTypes.ABS)
    # plotter.save_output_as_figure("outs/Target_" + str_current_datetime + ".png")

    # # Plot the input amplitude
    # plotter = Plotter(LightField(amp, phase, params.wavelength, params.pixel_size), output_type=PlotTypes.ABS)
    # plotter.save_output_as_figure("outs/Input_" + str_current_datetime + ".png")
    # Plot the result - output amplitude

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

    params.wavelength=wavelength_start

    for i in range(len(kernels)):
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
        result = trained_model([field, kernel]).numpy()
        result = result[0, 0, :, :] * np.exp(1j * result[0, 1, :, :])


        # Plot the result
        plotter = Plotter1(
        LightField.from_complex_array(result, params.wavelength, params.pixel_size)
        )
        plotter.save_output_amplitude("outs/Result_" + str(i) + "_" + str_current_datetime + ".bmp")
        # plotter.save_output_amplitude("outs/ResultNN.bmp")

        params.wavelength+=wavelength_step


