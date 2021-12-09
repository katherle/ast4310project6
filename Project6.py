import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.table import QTable  # to use tables with units
from astropy.modeling.models import BlackBody  # to compute the Planck function
from astropy.visualization import quantity_support
from matplotlib_inline.backend_inline import set_matplotlib_formats
from numpy import newaxis as nax  # to simplify the many uses of this
from mpl_toolkits.mplot3d import Axes3D # 3D plots

quantity_support()
set_matplotlib_formats('svg')
from matplotlib import cm
from cycler import cycler
plt.rc('legend', frameon=False)
plt.rc('figure', figsize=(7, 7 / 1.75)) # Larger figure sizes
plt.rc('font', size=12)

from scipy.integrate import cumtrapz   # for tau integration
from scipy.ndimage import shift  # for "rotating" 3D cubes
from scipy.special import wofz   # for Voigt function
from scipy.interpolate import interp1d # for interpolating tau_500

i_units = "kW m-2 sr-1 nm-1"  # More practical SI units

def read_table_units(filename):
    """
    Reads a table in a text file, formatted with column names in first row,
    and unit names on second row. Any deviation from this format will fail.
    """
    tmp = np.genfromtxt(filename, names=True)
    unit_names = open(filename).readlines()[1][1:].split()
    # Convert to astropy QTable to have units
    data = QTable(tmp)
    # Assign units to quantities in table, use SI units
    for key, unit in zip(data.keys(), unit_names):
        data[key].unit = unit
        data[key] = data[key].si  # We don't want to use deprecated units
    return data

def compute_hminus_cross_section(wavelength, temperature, electron_density):
    """
    Computes the H minus extinction cross section, both free-free and
    bound-free as per Gray (1992).

    Parameters
    ----------
    wavelength : astropy.units.quantity (array)
        Wavelength(s) to calculate in units of length.
    temperature: astropy.units.quantity (scalar or array)
        Gas temperature in units of K or equivalent.
    electron_density: astropy.units.quantity (scalar or array)
        Electron density in units of per cubic length.

    Returns
    -------
    extinction : astropy.units.quantity (scalar or array)
        Total H- extinction in si units.
        Shape: shape of temperature + (nwave,)
    """
    # Broadcast to allow function of temperature and wavelength
    temp = temperature[..., nax]
    wave = wavelength[nax]
    theta = 5040 * u.K / temp
    electron_pressure = electron_density[..., nax] * const.k_B * temp

    # Compute bound-free opacity for H-, following Gray 8.11-8.12
    sigma_coeff = np.array([2.78701e-23, -1.39568e-18,  3.23992e-14, -4.40524e-10,
                               2.64243e-06, -1.18267e-05,  1.99654e+00])
    sigma_bf = np.polyval(sigma_coeff, wave.to_value('AA'))
    sigma_bf = sigma_bf * 1.e-22 * u.m ** 2

    # Set to zero above the H- ionisation limit at 1644.4 nm
    sigma_bf[wave > 1644.2 * u.nm] = 0.

    # convert into bound-free per neutral H atom assuming Saha,  Gray p156
    k_const = 4.158E-10 * u.cm ** 2 / u.dyn
    gray_saha = k_const * electron_pressure.cgs * theta ** 2.5 * 10. ** (0.754 * theta)
    kappa_bf = sigma_bf * gray_saha                    # per neutral H atom

    # correct for stimulated emission
    kappa_bf *= (1 - np.exp(-const.h * const.c / (wave * const.k_B * temp)))

    # Now compute free-free opacity, following Gray 8.13
    # coefficients for 4th degree polynomials in the log of wavelength (in AA)
    coeffs = np.array([[-0.0533464, 0.76661, -1.685, -2.2763],
                          [-0.142631, 1.99381, -9.2846, 15.2827],
                          [-0.625151, 10.6913, -67.9775, 190.266, -197.789]])
    log_wave = np.log10(wave.to_value('AA'))
    log_theta = np.log10(theta.value)
    tmp = 0
    for i in range(3):
        tmp += np.polyval(coeffs[i], log_wave) * (log_theta ** i)
    kappa_ff = electron_pressure * (10 ** tmp)
    kappa_ff = kappa_ff * 1e-26 * (u.cm ** 4) / u.dyn

    return (kappa_bf + kappa_ff).si

#data
DATA_FILE = "qs006024_sap_s285.fits"
atm3d = QTable.read(DATA_FILE)
falc = read_table_units("falc.dat")

#rename variables
height = falc['height']
tau_500 = falc['tau_500']
m = falc['colmass']
temp = falc['temperature']
v_turb = falc['v_turb']
n_h = falc['hydrogen_density']
n_p = falc['proton_density']
n_e = falc['electron_density']
ptot = falc['pressure']
p_ratio = falc['p_ratio']
rho_tot = falc['density']


fig, ax = plt.subplots()
ax.plot(height.to('km'),temp, label='FALC')
# show column (100, 100):
ax.plot(atm3d['height'].to('km'), atm3d['temperature'][:, 100, 100], label='3D (100, 100)')
ax.set_ylim(4000, 10000)
ax.legend();

fig, ax = plt.subplots()
ax.imshow(atm3d['temperature'].value[-1])
#ax.imshow(atm3d['temperature'][-1])

alpha_H_3d = (compute_hminus_cross_section(500*u.nm, atm3d['temperature'], atm3d['electron_density'])[..., 0]*atm3d['hydrogen_density'])
alpha_T_3d = 6.652e-29*u.m**2 * atm3d['electron_density']
tau_500_3d = -cumtrapz(alpha_H_3d + alpha_T_3d, atm3d['height'], axis = 0, initial = 0)

n_hI = n_h - n_p
alpha_H_falc = (compute_hminus_cross_section(500*u.nm, temp, n_e)[..., 0]*n_hI)
alpha_T_falc = 6.652e-29*u.m**2 * n_e
tau_500_falc = -cumtrapz(alpha_H_falc + alpha_T_falc, height, initial = 0)

# x and y grids, from 0 to 6 Mm (box size), 256 points

X, Y = np.mgrid[0:6:256j, 0:6:256j]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# tau1_height must be a 2D array (256, 256) with the
# heights where tau = 1 in km
ax.plot_surface(X, Y, tau_500_3d[1], cmap='magma', rcount=100, ccount=100)
ax.axis((6.3, -0.3, -0.3, 6.3));
ax.set_xlabel("y (Mm)")
ax.set_ylabel("x (Mm)")
ax.set_zlabel("z (km)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
custom_cycler = cycler("color", cm.plasma(np.linspace(0, 0.95, 5)))
ax.set_prop_cycle(custom_cycler)
ax.plot(atm3d['height'], tau_500_3d[:, 20, 60])
ax.plot(atm3d['height'], tau_500_3d[:, 100, 89])
ax.plot(atm3d['height'], tau_500_3d[:, 19, 43])
ax.plot(atm3d['height'], tau_500_3d[:, 201, 4])
ax.plot(height, tau_500_falc, label='FALC')
#ax.set_xlim(-0.5e6, 0.5e6)
ax.set_yscale('log')
plt.grid()
plt.legend()
plt.show()

"""
points = (X, Y, atm3d['height'])

#print(points)
#print(interpn(points, tau_500_3d, points))

x = np.linspace(0, 255, 256)
y = np.linspace(0, 255, 256)
tau_func = RegularGridInterpolator((np.flip(atm3d['height']), x, y), tau_500_3d)
tau1_height = tau_func(np.array([0, x, y]))

"""

print(tau_500_3d.shape)

def tau_one(tau, height):
    nx,ny = tau_500_3d.shape[1:]
    tau1_height = np.zeros((nx,ny))

    for i in range(nx):
        for j in range(ny):
            inter = interp1d(tau[i,j,:], height)
            tau1_height[i,j] = inter(1.0)
    return tau1_height

#print(tau_one((tau_500_3d,0,-1), atm3d['height'])

tau1_height = tau_one(np.moveaxis(tau_500_3d,0,-1), atm3d['height'])*u.m

fig, ax = plt.subplots()
im = ax.imshow(tau1_height.value, cmap = 'magma')
fig.colorbar(im, ax=ax, shrink=0.6, label=r'Height where $\tau = 1$ [m]')
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# tau1_height must be a 2D array (256, 256) with the
# heights where tau = 1 in km
ax.plot_surface(X, Y, tau1_height, cmap='magma', rcount=100, ccount=100)
ax.axis((6.3, -0.3, -0.3, 6.3));
ax.set_xlabel("x (Mm)")
ax.set_ylabel("y (Mm)")
ax.set_zlabel("z (km)")
plt.tight_layout()
plt.show()
