import time
from datetime import datetime

import numpy as np

from lightprop.calculations import H_off_axis, H_on_axis, get_gaussian_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.nn import NN_FFTTrainer
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, Plotter1, PlotTypes

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 1024
    params.pixel_size = 0.1
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

    params.beam_diameter = 15
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

    # Build NNTrainer or NNMultiTrainer
    NN = NN_FFTTrainer()

    # Run NN optimization
    # In case of NNMultiTrainer provide kernel as 3rd argument.
    # Please try running different numbers of iterations (last parameter)
    # Check the difference in the output for different amounts of training


    kernels_number=2
    wavelength_start=PropagationParams.get_wavelength_from_nu(180)
    wavelength_stop=PropagationParams.get_wavelength_from_nu(180)

    wavelength_step=(wavelength_stop-wavelength_start)/(kernels_number-1)
    kernels=[None]*kernels_number

    params.wavelength=wavelength_start

    for i in range(len(kernels)):
        kernels[i]=np.array(
            [
                [
                    H_on_axis(
                        x / np.sqrt(x**2 + params.distance**2) / params.wavelength,
                        y / np.sqrt(params.distance**2 + y**2) / params.wavelength,
                        params.distance,
                        params.wavelength,
                    )
                    for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
                ]
                for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
        )
        kernels[i]=LightField.from_complex_array(kernels[i], params.wavelength, params.pixel_size)

        params.wavelength+=wavelength_step


    # kernel1 = np.array(
    #     [
    #         [
    #             H_off_axis(
    #                 x / np.sqrt(x**2 + params.distance**2) / params.wavelength,
    #                 y / np.sqrt(params.distance**2 + y**2) / params.wavelength,
    #                 params.distance,
    #                 params.wavelength,
    #             )
    #             for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #         ]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )

    # params.wavelength *= 1.1
    # kernel2 = np.array(
    #     [
    #         [
    #             H_off_axis(
    #                 x / np.sqrt(x**2 + params.distance**2) / params.wavelength,
    #                 y / np.sqrt(params.distance**2 + y**2) / params.wavelength,
    #                 params.distance,
    #                 params.wavelength,
    #             )
    #             for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #         ]
    #         for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
    #     ]
    # )

    # trained_model = NN.optimize(
    #     [
    #         LightField(amp, phase, params.wavelength, params.pixel_size),
    #         LightField(amp, phase, params.wavelength, params.pixel_size),
    #     ],
    #     [
    #         LightField(target, phase, params.wavelength, params.pixel_size),
    #         LightField(target, phase, params.wavelength, params.pixel_size),
    #     ],
    #     [
    #         LightField.from_complex_array(kernel1, params.wavelength, params.pixel_size),
    #         LightField.from_complex_array(kernel2, params.wavelength, params.pixel_size),
    #     ],
    #     params.distance,
    #     iterations=100,
    # )

  

    trained_model = NN.optimize(
        [LightField(amp, phase, params.wavelength, params.pixel_size)]*kernels_number,
        [LightField(target, phase, params.wavelength, params.pixel_size)]*kernels_number,
        kernels,
        params.distance,
        iterations=2000,
    )



    # Plot loss vs epochs
    NN.plot_loss()

    # Extract the optimized phase map from the trainable layer
    optimized_phase = np.array(trained_model.layers[3].get_weights()[0])

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    
    plotter = Plotter1(
        LightField(amp, optimized_phase, params.wavelength, params.pixel_size)
    )
    plotter.save_output_phase("outs/Structure_" + str_current_datetime + ".bmp")

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

        params.wavelength+=wavelength_step


    # time.sleep(1)

    # current_datetime = datetime.now()
    # str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    # kernel = np.array([np.real(kernel2), np.imag(kernel2)])
    # kernel = kernel.reshape(
    #     (
    #         1,
    #         2,
    #         params.matrix_size,
    #         params.matrix_size,
    #     ),
    #     order="F",
    # )

    # # Evaluate model on the input field
    # result = trained_model([field, kernel]).numpy()
    # result = result[0, 0, :, :] * np.exp(1j * result[0, 1, :, :])

    # # Plot the result
    # plotter = Plotter1(
    #     LightField.from_complex_array(result, params.wavelength, params.pixel_size)
    # )
    # plotter.save_output_amplitude("outs/Result_" + str_current_datetime + ".bmp")

