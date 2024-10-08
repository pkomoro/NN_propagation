import json
import logging


class ParamsValidationException(Exception):
    pass


class PropagationParams:
    c = 299792458

    def __init__(self, matrix_size, beam_diameter, focal_length, distance, pixel_size, wavelength):
        """
        Validates and converts propagation params.
        :param matrix_size: number of pixels on the side of square calculation matrix_size
        :param beam_diameter: sigma parameter of Gaussian beam in [mm]
        :param focal_length: focusing distance in [mm]
        :param distance: propagation distance in [mm]
        :param pixel_size: dimensions of the pixels used in calculations [mm]
        :param wavelength: wavelength of EM radiation in [mm]
        """
        logging.info("Loading propagation params")
        self.matrix_size = matrix_size
        self.wavelength = wavelength
        self.beam_diameter = beam_diameter
        self.focal_length = focal_length
        self.distance = distance
        self.pixel_size = pixel_size

    def __str__(self):
        return self.__dict__

    @property
    def matrix_size(self):
        return self._matrix_size

    @matrix_size.setter
    def matrix_size(self, size):
        self._matrix_size = self._positive_integer_validator(size)

    
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = self._positive_float_validator(value)

    @property
    def beam_diameter(self):
        return self._beam_diameter

    @beam_diameter.setter
    def beam_diameter(self, value):
        self._beam_diameter = self._positive_float_validator(value)

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = self._cast_to_type_validator(value, expected_type=int)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = self._cast_to_type_validator(value, expected_type=int)

    @property
    def pixel_size(self):
        return self._pixel

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel = self._positive_float_validator(value)

    def _positive_float_validator(self, value):
        return self._positive_value_validator(value, expected_type=float)

    def _positive_integer_validator(self, value):
        return self._positive_value_validator(value, expected_type=int)

    def _positive_value_validator(self, value, expected_type):
        value = self._cast_to_type_validator(value, expected_type)
        if expected_type(value) <= 0:
            raise ParamsValidationException(f"Value should be {expected_type} greater than 0")
        return value

    def _cast_to_type_validator(self, value, expected_type):
        try:
            return expected_type(value)
        except ValueError:
            raise ParamsValidationException(f"{value} cannot be converted to {expected_type}")

    @staticmethod
    def get_wavelength_from_frequency(nu):
        return PropagationParams.c / nu * 10**-6

    @classmethod
    def get_example_propagation_data(cls):
        data = {
            "matrix_size": 128,
            "wavelength": PropagationParams.get_wavelength_from_frequency(140),
            "beam_diameter": 20,
            "focal_length": 200,
            "distance": 200,
            "pixel_size": 1,
        }
        return cls.get_params_from_dict(data)

    @classmethod
    def get_params_from_dict(cls, params_dict):
        return cls(**params_dict)

    @classmethod
    def get_params_from_json_file(cls, json_file):
        with open(json_file) as file:
            data = json.load(file)
        return PropagationParams.get_params_from_dict(data)
