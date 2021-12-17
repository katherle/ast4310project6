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
from matplotlib.pyplot import cm

quantity_support()
set_matplotlib_formats('svg')
from matplotlib import cm
from cycler import cycler
plt.rc('legend', frameon=False)
plt.rc('figure', figsize=(7, 7 / 1.75)) # Larger figure sizes
plt.rc('font', size=12)
#from matplotlib import rcParams
#rcParams['axes.labelpad'] = 15

from scipy.integrate import cumtrapz   # for tau integration
from scipy.integrate import trapz   # for intensity integration
from scipy.ndimage import shift  # for "rotating" 3D cubes
from scipy.special import wofz   # for Voigt function
from scipy.interpolate import interp1d # for interpolating tau_500

i_units = u.Quantity(1, "kW m-2 sr-1 nm-1")  # More practical SI units

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




class Atom:
    """
    Reads atomic data, calculates level populations according to Boltzmann's law,
    and ionisation fractions according to Saha's law.
    """

    def __init__(self, atomfile=None):
        """
        Parameters
        ----------
        atomfile : string, optional
            Name of file with atomic data. If not present, atomic data needs
            to be loaded with the .read_atom method.
        """
        self.loaded = False
        if atomfile:
            self.read_atom(atomfile)

    def read_atom(self, filename):
        """
        Reads atom structure from text file.
        Parameters
        ----------
        filename: string
            Name of file with atomic data.
        """
        tmp = np.loadtxt(filename, unpack=True)
        self.n_stages = int(tmp[2].max()) + 1

        # Get maximum number of levels in any stage
        self.max_levels = 0
        for i in range(self.n_stages):
            self.max_levels = max(self.max_levels, (tmp[2] == i).sum())

        # Populate level energies and statistical weights
        # Use a square array filled with NaNs for non-existing levels
        chi = np.empty((self.n_stages, self.max_levels))
        chi.fill(np.nan)
        self.g = np.copy(chi)
        for i in range(self.n_stages):
            nlevels = (tmp[2] == i).sum()
            chi[i, :nlevels] = tmp[0][tmp[2] == i]
            self.g[i, :nlevels] = tmp[1][tmp[2] == i]

        # Put units, convert from cm-1 to Joule
        chi = (chi / u.cm).to('aJ', equivalencies=u.spectral())

        # Save ionisation energies, saved as energy of first level in each stage
        self.chi_ion = chi[:, 0].copy()

        # Save level energies relative to ground level in each stage
        self.chi = chi - self.chi_ion[:, np.newaxis]
        self.loaded = True

    def compute_partition_function(self, temperature):
        """
        Compute partition function using the atomic level energies and statistical  weights.
        Parameters:
        -------------
        temperature: astropy.units.quantity (scalar or array)
            Gas temperature in K or equivalent.
        """
        if not self.loaded:
            raise ValueError("Missing atom structure. Please load atom with read_atom() function.")

        temp = temperature[np.newaxis, np.newaxis]
        pfunc = np.nansum(self.g[..., np.newaxis]
                          *np.exp(-self.chi[..., np.newaxis]/const.k_B/temp), axis=1)
        return pfunc
    def compute_excitation(self, temperature):
        """
        Computes the level population relative to the ground state,
        according to Boltzmann's law.
        Parameters:
        ---------------
        temperature: astropy.units.quantity (scalar or array)
            Gas temperature in K or equivalent.
        """
        pfunc = self.compute_partition_function(temperature)

        #reshape arrays to allow broadcasting
        temp = temperature[np.newaxis, np.newaxis]
        g_ratio = self.g[..., np.newaxis]/pfunc[:, np.newaxis]
        chi = self.chi[..., np.newaxis]

        return g_ratio*np.exp(-chi/(const.k_B*temp))

    def compute_ionisation(self, temperature, electron_density):
        """
        Computes the ionisation fraction according to the Saha law.
        Parameters:
        ---------------
        temperature: astropy.units.quantity (scalar or array)
            Gas temperature in K or equivalent.
        electron_pressure: astropy.units.quantity (scalar)
            Electron pressure in Pa or equivalent.
        """
        partition_function = self.compute_partition_function(temperature)

        saha_const = ((2*np.pi*const.m_e*const.k_B*temperature)/const.h**2)**(3/2)
        nstage = np.zeros_like(partition_function)/u.m**3
        nstage[0] = 1./u.m**3

        for r in range(self.n_stages - 1):
            nstage[r+1] = (nstage[r]/electron_density * 2 * saha_const
                           * partition_function[r+1]/partition_function[r]
                           * np.exp(-self.chi_ion[r+1, np.newaxis]/(const.k_B*temperature[np.newaxis])))

        return nstage/np.nansum(nstage, axis = 0)

    def compute_populations(self, temperature, electron_density):
        """
        Computes relative level populations for all levels and all
        ionisation stages using the Boltzmann and Saha laws.
        Parameters
        ----------
        temperature: astropy.units.quantity (scalar or array)
            Gas temperature in units of K or equivalent.
        electron_pressure: astropy.units.quantity (scalar)
            Electron pressure in units of Pa or equivalent.
        """
        return (self.compute_excitation(temperature)
                * self.compute_ionisation(temperature, electron_density)[:, np.newaxis])


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


#Plot temperature vs height for FALC and 3D models
fig, ax = plt.subplots()
ax.plot(height.to('km'),temp, label='FALC')
# show column (100, 100):
ax.plot(atm3d['height'].to('km'), atm3d['temperature'][:, 100, 100], label='3D (100, 100)')
ax.set_ylim(4000, 10000)
ax.set_title('Comparison of temp vs height for 3D and FALC')
ax.legend()
plt.show()


#2D plot of temperature
fig, ax = plt.subplots()
im = ax.imshow(atm3d['temperature'].value[-1], cmap = 'coolwarm')
ax.set_title('Temperature at deepest point')
fig.colorbar(im, ax=ax, shrink=0.8, label=r'Temperature (K)')
plt.show()


#compute tau500 from the 3D model
alpha_H_3d = (compute_hminus_cross_section(500*u.nm, atm3d['temperature'], atm3d['electron_density'])[..., 0]*atm3d['hydrogen_density'])
alpha_T_3d = 6.652e-29*u.m**2 * atm3d['electron_density']
tau_500_3d = -cumtrapz(alpha_H_3d + alpha_T_3d, atm3d['height'], axis = 0, initial = 0)


#compute tau500 from the FALC model
n_hI = n_h - n_p
alpha_H_falc = (compute_hminus_cross_section(500*u.nm, temp, n_e)[..., 0]*n_hI)
alpha_T_falc = 6.652e-29*u.m**2 * n_e
tau_500_falc = -cumtrapz(alpha_H_falc + alpha_T_falc, height, initial = 0)


#set up x and y grids, from 0 to 6 Mm (box size), 256 points
X, Y = np.mgrid[0:6:256j, 0:6:256j]


#3D plot of tau500 from the 3D model
fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(projection='3d')
im = ax.plot_surface(X, Y, tau_500_3d[1], cmap='magma', rcount=100, ccount=100)
#fig.colorbar(im, ax=ax, shrink=0.6, label=r'$\tau_{500}$')
ax.axis((6.3, -0.3, -0.3, 6.3));
ax.set_xlabel("y (Mm)")
ax.set_ylabel("x (Mm)")
ax.set_zlabel(r"$\tau_{500}$ [1]", labelpad = 15)
ax.set_title(r"$\tau_{500}$ at highest point")
plt.tight_layout()
plt.show()

#plot of tau500 vs height for four randomly chosen columns + FALC
fig, ax = plt.subplots()
custom_cycler = cycler("color", cm.plasma(np.linspace(0, 0.95, 5)))
ax.set_prop_cycle(custom_cycler)
ax.plot(atm3d['height'].to('Mm'), tau_500_3d[:, 20, 60], label = '3D')
ax.plot(atm3d['height'].to('Mm'), tau_500_3d[:, 100, 89], label = '3D')
ax.plot(atm3d['height'].to('Mm'), tau_500_3d[:, 19, 43], label = '3D')
ax.plot(atm3d['height'].to('Mm'), tau_500_3d[:, 201, 4], label = '3D')
ax.plot(height.to('Mm'), tau_500_falc, label='FALC')
ax.set_yscale('log')
ax.set_xlabel('Height [Mm]')
ax.set_ylabel(r'$\tau_{500}$')
plt.grid()
plt.legend()
plt.show()



#define function to find the height at which tau=1
def tau_one(tau, height):
    nx,ny = tau_500_3d.shape[1:]
    tau1_height = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            inter = interp1d(tau[i,j,:], height)
            tau1_height[i,j] = inter(1.0)
    return tau1_height
tau1_height = tau_one(np.moveaxis(tau_500_3d,0,-1), atm3d['height'])*u.m
#plot height where tau=1 (2D)
fig, ax = plt.subplots()
im = ax.imshow(tau1_height.value, cmap = 'magma')
fig.colorbar(im, ax=ax, shrink=0.8, label=r'Height where $\tau_{500} = 1$ [m]')
plt.tight_layout()
plt.show()
#plot height where tau=1 (3D)
fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(projection='3d')
# tau1_height must be a 2D array (256, 256) with the
# heights where tau = 1 in km
ax.plot_surface(X, Y, tau1_height, cmap='magma', rcount=100, ccount=100)
ax.axis((6.3, -0.3, -0.3, 6.3));
ax.set_xlabel("x (Mm)")
ax.set_ylabel("y (Mm)")
ax.set_zlabel("z (km)", labelpad = 20)
ax.set_title(r"Height at which $\tau_{500} = 1$")
plt.tight_layout()
plt.show()
### Problem 2
#compute disk centre continuous intensity from 3D at mu=1
S = BlackBody(atm3d['temperature'], scale = i_units)(500*u.nm)
I = trapz(S*np.exp(-tau_500_3d), tau_500_3d, axis = 0).to(i_units)
#find intensity at 500 nm, mu=0.4 and 0.2
I_02 = trapz(S*np.exp(-tau_500_3d/0.2), tau_500_3d/0.2, axis = 0).to(i_units)
I_04 = trapz(S*np.exp(-tau_500_3d/0.4), tau_500_3d/0.4, axis = 0).to(i_units)
fig, ax = plt.subplots(2, 2, figsize = (10, 5))
im1 = ax[0, 0].imshow(I_02.value, cmap = 'inferno', aspect = 0.2)
ax[0, 0].set_title(r"$\mu=0.2$")
im2 = ax[1, 0].imshow(I_04.value, cmap = 'inferno', aspect = 0.4)
ax[1, 0].set_title(r"$\mu=0.4$")
gs = ax[0, 1].get_gridspec()
for axs in ax[:, -1]:
    axs.remove()
axbig = fig.add_subplot(gs[:, 1])
im3 = axbig.imshow(I.value, cmap = 'inferno', aspect = 1)
axbig.set_title(r"$\mu = 1$")
[fig.colorbar(imi, ax=axi, shrink = 0.6, label=lab) for imi, axi, lab in zip([im1, im2, im3], [ax[0, 0], ax[1, 0], axbig],
                                                                             ['Intensity', 'Intensity', r'Intensity [kW m$^{-2}$ sr$^{-1}$ nm$^{-1}$]'])]
plt.tight_layout()
plt.show()










### Problem 3


def damping(wave0, wave, P, T, d):
    """
    Finds the damping parameter for use with the Voigt function.
    wave0: Line center wavelength in nm
    wave: Array containing wavelengths in nm
    d: Doppler broadening in nm or equivalent
    P: Gas pressure in cgs units
    T: Temperature in K
    """
    #gamma (radiation)
    gamma_rad = (6.67e13*0.318/wave.to_value("nm")**2)/u.s

    #gamma (van der waals)
    n_u2 = const.Ryd.to('aJ', equivalencies = u.spectral()) * 1/(model_NaI.chi_ion[1] - model_NaI.chi[0, 1])
    n_l2 = const.Ryd.to('aJ', equivalencies = u.spectral()) * 1/(model_NaI.chi_ion[1] - model_NaI.chi[0, 0])

    r_u2 = n_u2/2 * (5*n_u2 + 1 - 3*2)
    r_l2 = n_l2/2 * (5*n_l2 + 1)

    log_vdw = 6.33 + 0.4*np.log10(r_u2 - r_l2) + np.log10(P.cgs.value) - 0.7*np.log10(T.si.value)

    gamma_vdw = (10**log_vdw)/u.s

    gamma = gamma_rad + gamma_vdw[..., nax]

    #damping equation:
    return wave**2/(4*np.pi*const.c) * gamma/d

def voigt(damping, wave_sep):
    """
    Calculates the Voigt function H(a, u),
    where a is the damping parameter and u is the separation from the line centre.
    Both parameters should be dimensionless.
    """
    z = (wave_sep + 1j * damping)
    return wofz(z).real

def doppler_3d(wave0, T):
    """
    Finds the Doppler broadening.
    wave0: Line center wavelength in nm
    T: Temperature in K
    """
    return wave0/const.c * np.sqrt(2*const.k_B*T/(22.99*const.u))

def compute_vlos_3d(wave0, v_x, v_y, v_z, mu, azimuthal_angle):
    """
    Computes the line of sight velocity .
    Parameters
    ----------
    wave0: astropy.units.quantity (scalar)
        Rest wavelength of the bound-bound transition, in units of length.
    v_x: astropy.units.quantity (scalar or array)
        Velocity field in x direction
    v_y: astropy.units.quantity (scalar or array)
        Velocity in y direction
    v_z: astropy.units.quantity (scalar or array)
        Velocity in z direction
    mu:  float (scalar)
        cos(polar angle)
    azimuthal_angle: float (scalar)
        Azimuthal angle
    Returns
    -------
    line of sight shift: astropy.units.quantity (scalar or array)
        line of sight shift in units of length. Same shape as velocities.
    """
    polar_angle = np.arccos(mu)
    v_los = mu*v_z + v_y*np.sin(polar_angle)*np.sin(azimuthal_angle) + v_x*np.sin(polar_angle)*np.cos(azimuthal_angle)

    return wave0.si / const.c * v_los


def intensity_func(wave0, wave, mu, T, N_E, N_HI, P, vx, vy, vz, z, azimuth, jump):
    """
    Calculates the intensity profiles
    wave0: Line center wavelength in nm
    wave: Array containing wavelengths in nm
    T: Temperature in K
    N_E: Electron density in m-3
    P: Gas pressure in cgs units
    vx, vy, vz: Velocity in the x, y, and z directions in m s-1
    mu: Viewing angle
    z: Height in m
    azimuth: Azimuth angle [unitless]
    jump: How manye columns should be skipped for each calculated column. Example jump=8 gives 32 columns.
    """

    I = np.zeros((len(lam),T.shape[1]//jump, T.shape[2]//jump))*i_units
    I_c = np.zeros((len(lam),T.shape[1]//jump, T.shape[2]//jump))*i_units
    #print(I.shape)

    for i in range(T.shape[1]//jump):
    #for i in range(1):
        for j in range(T.shape[2]//jump):
        #for j in range(1):
            ANa = 1.7378e-6  #Na abundance
            constant = (const.e.si**2/(4*const.eps0*const.m_e*const.c) * N_HI[:,i,j] * ANa * 0.318)[..., nax]

            #stimulated emission

            stim = (1 - np.exp(-const.h*const.c/(wave*const.k_B*T[:,i,j,nax])))

            #level population and line profile

            v_los_shift = compute_vlos_3d(wave0, vx[:,i,j], vy[:,i,j], vz[:,i,j], mu, azimuth)[..., nax]
            doppler_width = doppler_3d(wave0, T[:,i,j])[..., nax]
            level_pop = model_NaI.compute_populations(T[:,i,j], N_E[:,i,j])[0,0][..., nax]


            a = damping(wave0, wave, P[:,i,j], T[:,i,j], doppler_width)
            v = ((wave - wave0 + v_los_shift) / doppler_width).si

            profile = voigt(a, v)/(np.sqrt(np.pi)*doppler_width)


            alpha_Hminus = compute_hminus_cross_section(wave, T[:,i,j,nax], N_E[:,i,j,nax])[..., 0]*N_HI[:,i,j,nax]
            alpha_T = 6.652e-29*u.m**2 * N_E[:,i,j,nax]
            alpha_li = constant * wave**2 * level_pop * profile * stim / const.c
            alpha_tot =  alpha_li + alpha_Hminus + alpha_T
            #print(alpha_tot.shape)

            alpha_cont = alpha_Hminus + alpha_T

            #optical depth
            tau = -cumtrapz(alpha_tot.decompose(), z, axis = 0, initial = 0) #got rid of negative sign bc height is flipped

            #source function
            S = BlackBody(T[:,i,j,nax], scale = i_units)(wave)
            #total intensity
            I[:,i,j] = trapz(S*np.exp(-tau/mu), tau/mu, axis = 0)

            #continuum intensity (for normalization)
            tau_c = -cumtrapz(alpha_cont, z, axis = 0, initial = 0)
            I_c[:,i,j] = trapz(S*np.exp(-tau_c/mu), tau_c/mu, axis = 0)

        print(i)


    return I, I_c


model_NaI = Atom("NaI_atom.txt")
wave0 = model_NaI.chi[0, 1].to("nm", equivalencies = u.spectral())
lam = np.linspace(585,595,200)*u.nm


#intensity = intensity_func(wave0, lam, 1., atm3d['temperature'], atm3d['electron_density'], atm3d['hydrogen_density'], atm3d['pressure'], atm3d['velocity_x'], atm3d['velocity_y'], atm3d['velocity_z'], atm3d['height'], 0, 1)


#I = intensity[0]
#I_c = intensity[1]

#np.save('I.npy', I.value)
#np.save('I_c.npy', I_c.value)

I = np.load('I.npy')
I_c = np.load('I_c.npy')


I_mean = I.mean(axis=(1,2))

color = iter(cm.viridis(np.linspace(0, 1, 32)))

plt.grid()
for i in range(32):
    c = next(color)
    plt.plot(lam,I[:,8*i,8*i], c=c)
plt.plot(lam,I_mean, '--', color='red', linewidth = 5.0,  label='Average')

plt.xlabel(r'$\lambda$ [nm]', fontsize=14)
plt.ylabel(r'$|I_\lambda|$ [$\mathrm{kW nm^{-1} sr^{-1} m^{-2}}$]', fontsize=14)
plt.title('Intenstiy for a selction of columns', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('some_columns.pdf', dpi=900)
plt.show()

color = iter(cm.viridis(np.linspace(0, 1, 256)))

plt.grid()
for i in range(len(lam)):
    c = next(color)
    plt.plot(lam,I[:,i,i], c=c)
plt.plot(lam,I_mean, '--', color='red', linewidth = 5.0, label='Average')

plt.xlabel(r'$\lambda$ [nm]', fontsize=14)
plt.ylabel(r'$|I_\lambda|$ [$\mathrm{kW nm^{-1} sr^{-1} m^{-2}}$]', fontsize=14)
plt.title('Intenstiy for all columns', fontsize=16)
plt.legend(fontsize=14)
plt.savefig('all_columns.pdf', dpi=900)
plt.show()


# 3.2

I_swap = np.swapaxes(I, 2,0)
print(I_swap.shape)
plt.imshow(I_swap[:,0], cmap='magma', extent=[lam[0].value, lam[-1].value, 0, 6])
plt.ylabel('y [Mm]', fontsize=14)
plt.xlabel(r'$\lambda$ [nm]', fontsize=14)
cbar = plt.colorbar()
cbar.set_label(r'$|I_\lambda|$ [$\mathrm{kW nm^{-1} sr^{-1} m^{-2}}$]', fontsize=14)
plt.title('Spectrograph of a slice of 3D space')
plt.savefig('spectrograph.pdf', dpi=900)
plt.show()


I_stronk_arg = np.argwhere(I[0,:,:] == np.min(I[0,:,:]))
I_weak_arg = np.argwhere(I[0,:,:] == np.max(I[0,:,:]))
print(I_stronk_arg)
print(I_weak_arg)

plt.plot(lam, I[:,239,181])
plt.plot(lam, I[:,44,71])
plt.xlabel(r'$\lambda$ [nm]', fontsize=14)
plt.grid()
plt.ylabel(r'$|I_\lambda|$ [$\mathrm{kW nm^{-1} sr^{-1} m^{-2}}$]', fontsize=14)
plt.title('Intensity for weak and strong line core', fontsize=16)
plt.savefig('weak_and_strong.pdf', dpi=900)
plt.show()




# 3.3


def Flux_calc(wave0, wave, mu, T, N_E, N_HI, P, vx, vy, vz, z, azimuth, jump):
    """
    Calculates the 3D Flux for 4 values of azimuth.
    wave0: Line center wavelength in nm
    wave: Array containing wavelengths in nm
    T: Temperature in K
    N_E: Electron density in m-3
    P: Gas pressure in cgs units
    vx, vy, vz: Velocity in the x, y, and z directions in m s-1
    mu: Viewing angle
    z: Height in m
    azimuth: Azimuth angle [unitless]
    jump: How manye columns should be skipped for each calculated column. Example jump=8 gives 32 columns.
    """

    f = np.zeros((4,3,len(lam),T.shape[1]//jump, T.shape[2]//jump))*i_units
    f_c = np.copy(f)

    weight_gauss = (np.array([5/9, 8/9, 5/9]) / 2)

    for l in range(len(azimuth)):
        for k in range(len(mu_gauss)):
            I_all = intensity_func(wave0, lam, mu[k], atm3d['temperature'], atm3d['electron_density'], atm3d['hydrogen_density'], atm3d['pressure'], atm3d['velocity_x'], atm3d['velocity_y'], atm3d['velocity_z'], atm3d['height'], azimuth[l], jump)
            I_flux = I_all[0]
            I_continuum = I_all[1]
            f[l,k,...] = (weight_gauss[k]*mu_gauss[k]*I_flux[nax,nax,...])
            f_c[l,k,...] = (weight_gauss[k]*mu_gauss[k]*I_continuum[nax,nax,...])

        flux = np.sum(f,axis=1)
        flux_continuum = np.sum(f_c, axis=1)

        print(l)

    return flux, flux_continuum


mu_gauss = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]) / 2 + 0.5
azimuth = np.array([0,1,2,3])


"""
#This is commented out, as it has been run before, and the calculation takes some time.

Fluxes = Flux_calc(wave0, lam, mu_gauss, atm3d['temperature'], atm3d['electron_density'], atm3d['hydrogen_density'], atm3d['pressure'], atm3d['velocity_x'], atm3d['velocity_y'], atm3d['velocity_z'], atm3d['height'], azimuth, 8)

Flux = Fluxes[0]
Flux_c = Fluxes[1]

np.save('Flux.npy', Flux.value)
np.save('Flux_c.npy', Flux_c.value)
"""

# Loading the pre calculated values.
Flux = np.load('Flux.npy')
Flux_c = np.load('Flux_c.npy')

# Averageing over all azimuth angles.
Flux = np.mean(Flux, axis=0)
# Spatially averaging
mean_flux = np.mean(Flux, axis=(1,2))

plt.plot(lam,abs(mean_flux))
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'$F^+$ [$\mathrm{kW nm^{-1} sr^{-1} m^{-2}}$]')
plt.title(r'$F^+$ as a function of $\lambda$ for all angles')
plt.grid()
plt.savefig('flux.pdf', dpi=900)
plt.show()
