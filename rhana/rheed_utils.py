import numpy as np

def hv2wl(voltage):
    """
    Convert accelerating voltage to electron wavelength
    Args:
        voltage (float): acceleration voltage of electron in kV

    Returns:
        wavelength in mm
    """
    # me = 510998.946 # in electron volts / (the speed of light^2)
    me = 9.10938356e-31 # in kilograms
    # h = 4.1357e-15 # in eV s
    h = 6.62607004e-34 #in m2 kg / s    
    
    E = voltage * 1e3 * 1.6022e-19 # convert keV to J
    wave_length = h / (me * np.sqrt( 2 * E / me ) ) * 1e3 # use matter wave equation to convert kinetic energy to momentum and wavelength
    
    return wave_length


def wl2k(wl):
    return 2*np.pi / wl 