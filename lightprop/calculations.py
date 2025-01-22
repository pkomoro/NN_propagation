import numpy as np

from lightprop.propagation.params import PropagationParams


# impulse response function in the on-axis approximation
def h(r, distance, wavelength):
    return (
        np.exp(1j * 2 * (np.pi / wavelength) * distance)
        / (1j * wavelength * distance)
        * np.exp(1j * np.pi / (wavelength * distance) * r * r)
    )


# transfer function in the on-axis approximation
def H_on_axis(vx, vy, distance, wavelength):
    return np.exp(1j * 2 * (np.pi / wavelength) * distance) * np.exp(
        -1j * np.pi * wavelength * distance * (vx**2 + vy**2)
    )


# full transfer function
def H_off_axis(vx, vy, distance, wavelength):
    return np.exp(
        1j * 2 * (np.pi / wavelength) * distance * np.sqrt(1 - (vx * wavelength) ** 2 - (vy * wavelength) ** 2)
    )


def gaussian(r, variance):
    return np.exp(-(r**2) / (2 * variance**2))


def lens(r, focal_length, wavelength):
    return (-2 * np.pi) / wavelength * np.sqrt(r**2 + focal_length**2)

def FZP_phase(r, rs):
    counter = 0
    for i in range(len(rs)):
        if r >= rs[i]:
            counter+=1
    if counter % 2 == 0:
        return 0
    else:
        return np.pi



def get_FZP_distribution(params: PropagationParams, x0: float = 0, y0: float = 0):
    zones = 20
    rs = [0] * zones
    for i in range(zones):
        rs[i] = np.sqrt(2*params.focal_length / 2 * i * params.wavelength)

    return np.array(
        [
            [
                FZP_phase(np.sqrt((x-x0)**2 + (y-y0)**2), rs)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )


def compare_np_arrays(array1, array2):
    return np.max(array1 - array2) < 10**-6


def get_lens_distribution(params: PropagationParams, x0: float = 0, y0: float = 0):
    return np.array(
        [
            [
                lens(np.sqrt((x-x0)**2 + (y-y0)**2), params.focal_length, params.wavelength)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )


def get_gaussian_distribution(params: PropagationParams, x0: float = 0, y0: float = 0):
    return np.array(
        [
            [
                gaussian(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), params.beam_diameter)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

def get_flat_array(params: PropagationParams, A: float = 0):
    return np.array(
        [
            [
                A
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )


def circle(x,y,r,fill):
    if np.sqrt(x**2 + y**2) <= r:
        return fill
    else:
        return 1-fill


def circle_aperture(params: PropagationParams, r: float, fill: float):
    return np.array(
        [
            [
                circle(x, y, r, fill)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )
