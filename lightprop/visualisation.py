import logging
import os
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt

from lightprop.lightfield import LightField


# TODO enum?
class PlotTypes:
    INTENSITY = "intensity"
    PHASE = "phase"
    ABS = "abs"


class Plotter:
    def __init__(self, propagation_result: LightField, output_type=PlotTypes.ABS):
        self.propagation_result = propagation_result
        logging.info("Plotting image data")
        plot_type = {
            PlotTypes.ABS: self.propagation_result.get_amplitude,
            PlotTypes.INTENSITY: self.propagation_result.get_intensity,
            PlotTypes.PHASE: self.propagation_result.get_phase_scaled,
        }
        self.data = plot_type[output_type]()

        plt.imshow(self.data, interpolation="nearest")

    def save_output_as_figure(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.savefig(path)
        logging.info("Generated")

    def save_output_as_bitmap(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.imsave(path,self.data, cmap='gray')
        logging.info("Generated")

    def show(self):
        plt.show()

    def _prepare_path_to_save(self, path):
        logging.info("Preparing directories")
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)


class Plotter1:
    def __init__(self, propagation_result: LightField):
        self.propagation_result = propagation_result
        logging.info("Plotting image data")
        

    def save_output_amplitude(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.imsave(path,self.propagation_result.get_amplitude(), cmap='gray')
        logging.info("Generated")

    def save_output_intensity(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.imsave(path,self.propagation_result.get_intensity(), cmap='gray')
        logging.info("Generated")

    def save_output_phase(self, path):
        self._prepare_path_to_save(path)
        logging.info(f"Saving to {path}")
        plt.imsave(path,np.mod(self.propagation_result.get_phase(), 2*np.pi), cmap='gray', vmin=0, vmax=2*np.pi)
        logging.info("Generated")

    def show(self):
        plt.show()

    def _prepare_path_to_save(self, path):
        logging.info("Preparing directories")
        dirs = os.path.dirname(path)
        Path(dirs).mkdir(parents=True, exist_ok=True)
