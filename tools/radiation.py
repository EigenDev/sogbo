#! /usr/bin/env python


import numpy as np 
import astropy.constants as const 
import astropy.units as units 
import matplotlib.pyplot as plt 
import utility as util 
import argparse
import os 

# FONT SIZES
SMALL_SIZE   = 8
DEFAULT_SIZE = 10
BIGGER_SIZE  = 12

#================================
#   constants of nature
#================================
# R_0 = const.R_sun.cgs 
# c   = const.c.cgs
# m   = const.M_sun.cgs
 
# rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
# e_scale    = m * c **2
# pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
# time_scale = R_0 / c


#==============
# BMK Scales
#==============
e_scale      = 1e53 * units.erg 
e0           = 1.0 
rho0         = 1.0
rho_scale    = 1.0 * const.m_p.cgs / units.cm**3 
temp_scale   = (1.e-10 * units.Kelvin * const.k_B.cgs).to(units.erg)
length_scale = ((e0 * e_scale / (rho0 * rho_scale * const.c.cgs**2))**(1/3)).to(units.cm)
time_scale   = length_scale / const.c.cgs 
pre_scale    = e_scale / length_scale ** 3

from collections import deque
from bisect import insort, bisect_left
from itertools import islice
def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def shock_finder(vfield: np.ndarray, r: np.ndarray) -> int:
    """
    Computes the divergence of the vector field vfield, corresponding to dFx/dx + dFy/dy + ...
    :return: index where divergence becomes negative
    """
    def first_neg(arr: np.ndarray):
        idx = 0
        for number in arr:
            idx += 1 
            if number < 0:
                return idx
    
    n = 5
    sections = vfield.size // n
    num_dims = np.ndim(vfield)
    divs     = []
    for nth in range(sections):
        edge  = nth + n if nth + 5 < vfield.size - 1 else  vfield.size - 1
        div_v = 1/r[nth:edge]**2 * np.ufunc.reduce(np.add, np.gradient(r[nth:edge]**2*vfield[nth:edge], r[nth:edge]))
        divs += [div_v]
    
    divs   = np.asanyarray(divs)

def calc_bfield_shock(rho: float, lorentz_gamma: float, beta: float, eps_b: float = 0.1) -> float:
    """
    Calc magnitude of magnetic field assuming shock jump conditions
    
    Params:
    ------------------------------------------------
    rho:           fluid density in rest frame 
    lorentz_gamma: lorentz factor 
    beta:          dimensionless velocity of flow 
    eps_b:            magnetic energy fraction
    
    Return:
    ------------------------------------------------
    The magnetic field magnitude 
    """
    return (8.0 * np.pi * eps_b * rho * rho_scale)**0.5 * lorentz_gamma * beta * const.c.cgs

def calc_bfield_shock_other(einternal: float, eps_b: float) -> float:
    """
    Calc magnitude of magnetic field assuming shock jump conditions
    
    Params:
    ------------------------------------------------
    einternal:     internal energy density
    eps_b:            magnetic energy fraction
    
    Return:
    ------------------------------------------------
    The magnetic field magnitude 
    """
    return (8.0 * np.pi * eps_b * einternal)**0.5
    
def calc_gyration_frequecy(b_field: float) -> float:
    """
    return gyration frequency in units of MHz
    Params
    ---------------------
    b_field: magnetic field value 
    
    Return
    ---------------------
    the gyration frequency 
    """
    frequency_for_unit_field = (const.e.gauss * 1 * units.gauss) / (2.0 * np.pi * const.m_e.cgs * const.c.cgs)
    return frequency_for_unit_field.value  * b_field.value * units.Hz

def calc_total_synch_power(lorentz_gamma: float, ub: float, beta: float) -> float:
    """
    Params:
    --------------------------------------
    lorentz_gamma:   lorentz factor 
    ub:              magnetic field energy density 
    beta:            dimensionless flow veloccity 
    
    Return:
    --------------------------------------
    Total synchrotron power
    """
    return (4.0/3.0) * const.sigma_T.cgs * beta**2 * lorentz_gamma ** 2 * const.c.cgs * ub

def calc_nphotons_per_bin(volume: float, n_e: float, nu_g:float, gamma_e: float, beta: float, u_b: float, dt: float, p: float = 2.5) -> float:
    """
    Calculate the number of photons per energy (gamma_e) bin
    
    Params:
    ------------------------------------------
    volume:              volume of the cell
    n_e:                 number density of electrons
    nu_g:                gyration frequency of electrons
    gamma_e:             lorentz factor of electrons
    u_b:                 magnetic field energy density 
    dt:                  the checkpoint time interval 
    p:                   electron spectral index 
    
    Return:
    ------------------------------------------
    number of photons per energy bin
    """
    a = (8.0 * np.pi * volume / (3.0 * const.h.cgs * nu_g))
    b = const.sigma_T.cgs * const.c.cgs * beta**2 * u_b * n_e 
    c = gamma_e**(-(p + 1.0))
    return a * b * c * dt

# def calc_nphotons_per_electron(volume: float, n_e: float, nu_g:float, nu_c: float, beta: float, u_b: float, dt: float, p: float = 2.5) -> float:
#     """
#     Calculate the number of photons in energy bin
    
#     Params:
#     ------------------------------------------
#     volume:              volume of the cell
#     n_e:                 number density of electrons
#     nu_g:                gyration frequency of electrons
#     nu_c:                critical frequency of photons
#     beta:                dimensionless velocity wrt to speed of light
#     u_b:                 magnetic field energy density 
#     dt:                  the checkpoint time interval 
#     p:                   electron spectral index 
    
#     Return:
#     ------------------------------------------
#     number of photons per energy bin
#     """
#     a = (16.0 * np.pi * volume / (3.0 * const.h.cgs * nu_g * (p - 1)))
#     b = const.sigma_T.cgs * const.c.cgs * beta**2 * u_b * n_e 
#     c = (nu_c/nu_g)**(-(p - 1.0)/2.0)
#     return a * b * c * dt

def calc_nphotons_other(power: float, nu_c: float, dt:float):
    """
    Calculate the number of photons per electron in energy bin
    
    Params: 
    -------------------------------------
    power:                 the total synchrotron power per electron
    nu_c:                  the critical frequency of electrons
    dt:                    the time step size between checkpoints
    
    Return:
    -------------------------------------
    calculate the number of photons emitted per electron in some energy bin
    """
    return power * dt / (const.h.cgs * nu_c)

def gen_random_from_powerlaw(a, b, g, size=1):
    """Power-law generator for pdf(x) ~ x^{g-1} for a<=x<=b"""
    rng = np.random.default_rng()
    r = rng.random(size=size)[0]
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1.0/g)

def vector_magnitude(a: np.ndarray) -> np.ndarray:
    """return magnitude of vector(s)"""
    if a.ndim <= 3:
        return (a.dot(a))**0.5
    else:
        return (a[0]**2 + a[1]**2 + a[2]**2)**0.5
        
def vector_dotproduct(a: np.ndarray, b: np.ndarray) -> float:
    """dot product between vectors or array of vectors"""
    if a.ndim <= 3:
        return a.dot(b)
    else:
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def calc_nelectrons_at_gamma_e(n_e: float, gamma_e: float, p: float = 2.5) -> float:
    """
    Calculate the number density of electrons per lorentz facor 
    Params:
    --------------------------------------
    n_e:                       total number density of electrons
    gamma_e:                   the electron lorentz factor
    dgamma:                    the energy bin size 
    p:                         electron spectral index 
    
    Return:
    -------------------------------------
    number density of electrons at a given energy
    """
    return n_e * gamma_e ** (-p)

def calc_doppler_delta(lorentz_gamma: float, beta_vector: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """
    Calculate the dpppler factor given lorentz factor, velocity vector, and propagation vector
    
    Params:
    --------------------------------------
    lorentz_gamma:           lorentz factor of flow 
    beta_vector:             velocity vector of flow 
    n_hat:                   direction of emission
    
    Return:
    ---------------------------------------
    the standard doppler factor delta 
    """
    return 1.0 / (lorentz_gamma * (1.0  - vector_dotproduct(beta_vector, n_hat)))

def calc_nu(gamma_e: float, nu_g: float):
    """Calculate frequency as function of lorentz_factor"""
    return gamma_e ** 2 * nu_g 

def calc_critical_lorentz(bfield: float, time: float)-> float:
    """Calculate the critical Lorentz factor as function of time and magnetic field"""
    return (6.0 * np.pi * const.m_e.cgs * const.c.cgs) / (const.sigma_T.cgs * bfield ** 2 * time)

def calc_max_power_per_frequency(bfield: float) -> float:
    """Calculate the maximum power per frequency""" 
    return (const.m_e.cgs * const.c.cgs ** 2 * const.sigma_T.cgs) / (3.0 * const.e.gauss) * bfield

def calc_gamma_min(eps_e: float, p: float) -> float:
    """
    Calculate the minimum lorentz factor of electrons in the distribution
    
    Params:
    ------------------------------
    eps_e:              fraction of internal energy due to electric field
    p:                  spectral electron number index 
    
    Return:
    ------------------------------
    Minimum lorentz factor of electrons
    """
    return eps_e * (p - 2.0)/ (p - 1.0) * const.m_p.cgs / const.m_e.cgs 

def sari_piran_narayan_99(
    fields:         dict, 
    args:           argparse.ArgumentParser, 
    frequency:      float,
    discrete_times: np.ndarray,
    flux_array:     np.ndarray,
    mesh:           dict, 
    dset:           dict, 
    past:           dict,
    overplot:       bool=False, 
    subplot:        bool=False, 
    ax:             plt.Axes=None, 
    case:           int=0
) -> None:
    
    def flux(
        flux_max: float, 
        p:        float,
        nu:       float, 
        nu_c:     float, 
        nu_m:     float, 
        ndim:     int = 1, 
        cooling:  str = 'slow') -> float:
        """
        ---------------------------------------
        Compute the flux according to https://arxiv.org/abs/astro-ph/9712005
        ---------------------------------------
        """
        fluxes = np.zeros_like(flux_max)
        if ndim == 1:
            for ridx in range(r.size):
                nu_crit = nu_c[ridx]
                nu_min  = nu_m[ridx]
                # Slow cooling
                if cooling == 'slow':
                    if nu_min > nu:
                        fluxes[:, :, ridx] = (nu/nu_min)**(1/3)*flux_max[:, :, ridx]
                    elif (nu < nu_crit) and (nu > nu_min):
                        fluxes[:, :, ridx] = (nu/nu_min) **(-(p-1)/2)*flux_max[:, :, ridx]
                    elif nu > nu_crit:
                        fluxes[:, :, ridx] = (nu_crit/nu_min)**(-(p-1)/2)*(nu/nu_crit)**(-p/2)*flux_max[:, :, ridx] 
                else:
                # Fast Cooling
                    if nu_crit > nu:
                        fluxes[:, :, ridx] = (nu/nu_crit)**(1/3)*flux_max[:, :, ridx]
                    elif (nu < nu_min) and (nu > nu_crit):
                        fluxes[:, :, ridx] = (nu/nu_crit) **(-1/2)*flux_max[:, :, ridx]
                    elif nu > nu_min:
                        fluxes[:, :, ridx] = (nu_min/nu_crit)**(-1/2)*(nu/nu_min)**(-p/2)*flux_max[:, :, ridx] 
        else:
            for tidx in range(theta.size):
                for ridx in range(r.size):
                    nu_crit = nu_c[tidx, ridx]
                    nu_min  = nu_m[tidx, ridx]
                    # Slow cooling
                    if cooling == 'slow':
                        if nu_min > nu:
                            fluxes[:, tidx, ridx] = (nu/nu_min)**(1/3)*flux_max[:, tidx, ridx]
                        elif (nu < nu_crit) and (nu > nu_min):
                            fluxes[:, tidx, ridx] = (nu/nu_min) **(-(p-1)/2)*flux_max[:, tidx, ridx]
                        elif nu > nu_crit:
                            fluxes[:, tidx, ridx] = (nu_crit/nu_min)**(-(p-1)/2)*(nu/nu_crit)**(-p/2)*flux_max[:, tidx, ridx] 
                    else:
                    # Fast Cooling
                        if nu_crit > nu:
                            fluxes[:, tidx, ridx] = (nu/nu_crit)**(1/3)*flux_max[:, tidx, ridx]
                        elif (nu < nu_min) and (nu > nu_crit):
                            fluxes[:, tidx, ridx] = (nu/nu_crit) **(-1/2)*flux_max[:, tidx, ridx]
                        elif nu > nu_min:
                            fluxes[:, tidx, ridx] = (nu_min/nu_crit)**(-1/2)*(nu/nu_min)**(-p/2)*flux_max[:, tidx, ridx] 
        return fluxes 
    
    beta       = util.calc_beta(fields)
    w          = util.calc_lorentz_gamma(fields)
    t_prime    = dset['time'] * time_scale
    t_emitter  = t_prime / w 
    #================================================================
    #                    HYDRO CONDITIONS
    #================================================================
    p     = 2.5  # Electron number index
    eps_b = 0.01 # Electric field fraction of internal energy 
    eps_e = 0.1  # Magnetic field fraction of internal energy
    
    rho_einternal = fields['p'] * pre_scale / (dset['ad_gamma'] - 1.0)   # internal energy density
    bfield        = calc_bfield_shock_other(rho_einternal, eps_b)        # magnetic field based on equipartition
    ub            = bfield**2 / (8.0 * np.pi)                            # magnetic energy density
    n_e           = fields['rho'] * w * rho_scale / const.m_p.cgs        # electron number density
    dt            = args.dt * time_scale                                  # step size between checkpoints
    nu_g          = calc_gyration_frequecy(bfield)                        # gyration frequency
    
    gamma_min  = w * calc_gamma_min(eps_e, p)               # Minimum Lorentz factor of electrons 
    gamma_max  = args.gamma_lims[1]                         # Maximum Lorentz factor of electrons
    gamma_crit = calc_critical_lorentz(bfield, t_emitter)   # Critical Lorentz factor of electrons

    radial_cell = beta.argmax()
    r = mesh['r']
    angular_samples = 10
    ndim = 1 
    if 'theta' in mesh:
        ndim += 1
        theta = mesh['th']
    else:
        theta = np.linspace(0.0, np.pi, angular_samples)
        
    if 'phi' in mesh:
        ndim += 1
        phi  = mesh['phi']
    else:
        phi = np.linspace(0.0, 2.0 * np.pi, angular_samples)

    thetta, phii, rr = np.meshgrid(theta, phi, r)
    
     # Calc cell volumes
    if ndim == 1:
        dvolume = util.calc_cell_volume1D(mesh['r']) * length_scale**3  
    elif ndim == 2:
        dvolume = util.calc_cell_volume2D(mesh['rr'], mesh['theta']) * length_scale**3 
        
    rhat     = np.array([np.sin(thetta)*np.cos(phii), np.sin(thetta)*np.sin(phii), np.cos(thetta)])  # radiail diirectional unit vector                                                                                 # electron particle number index   
    beta_vec = beta * rhat
    
    # Place observer along chosen axis
    theta_obs  = 0.0 * np.ones_like(thetta)
    obs_hat    = np.array([np.sin(theta_obs)*np.cos(phii), np.sin(theta_obs) * np.sin(phii), np.cos(theta_obs)])
    
    # We choose a basis such that the observer lies along the direction nhat
    # retarded time transformation
    d       = (4.0 * units.Gpc).to(units.cm)
    dt_obs = t_prime - past['t_prime'] if case != 0 else dt # time since observation n-1
    
    if case == 0:
        t_obs   = t_prime - rr * length_scale * vector_dotproduct(rhat, obs_hat) / const.c.cgs
    else:
        t_obs = past['t_obs'] + dt_obs
    
    # Calculate the maximum flux based on the average bolometric power per electron
    nu_c              = w * calc_nu(gamma_crit, nu_g)                                   # Critical frequency
    nu_m              = w * calc_nu(gamma_min, nu_g)                                    # Minimum frequency
    n_e               = fields['rho'] * w * rho_scale / const.m_p.cgs                   # electron number density
    dvolume           = util.calc_cell_volume1D(mesh['r']) * length_scale**3            # volume differential for a 1D run
    electron_number   = n_e * dvolume                                                   # Total number of electrons in run
    alpha             = (p - 1.0)/2.0                                                   # spectral index
    delta_doppler     = calc_doppler_delta(w, beta_vector=beta_vec, n_hat=obs_hat)      # Doppler factor
    nelectrons        = calc_nelectrons_at_gamma_e(electron_number, gamma_crit)         # electrons at the critical frequency
    one_power         = w * calc_max_power_per_frequency(bfield)                        # Synchrotron power of one electron
    flux_max          = nelectrons * one_power * delta_doppler ** (3.0 + alpha)         # Maximum flux 
                                  #
    ff = flux(flux_max, p, frequency * units.Hz, nu_c, nu_m, ndim = ndim)
    ff = dt / dt_obs * (ff / (4.0 * np.pi * d**2)).to(1e-6 * units.Jy)

    theta_idx = util.find_nearest(theta, 0.0)[0]
    if case == 0:
        past['t0']  = t_obs[:, theta_idx,:].max()
        
    past['t_obs']   = t_obs
    past['t_prime'] = t_prime
        
    # print("t = {:.2e}, dt = {:.2e}, dt/dt_obs = {:.2e}".format(t_prime, dt_obs, dt/dt_obs))
    # beam = t_obs[0][0][:30]
    # print(beam[beam > 0].min())
    # print("time: ", t_obs[0][0][:30].to(units.day))
    # print("flux: ", ff[0][0][:30])
    # zzz = input('')
    
    discrete_times.append(t_obs.value)
    flux_array.append(ff.value)
    
def log_events(
    fields:        dict, 
    args:          argparse.ArgumentParser, 
    events_list:   list,
    mesh:          dict, 
    dset:          dict, 
    case:          int=0) -> None:
    """
    Log the photon events based on synchrotron radiaiton (no asborption)
    
    TODO: Check against Blandford-McKee solution 
    Params:
    fields:           dictionary of fluid variables
    args:             argparse parser
    events_list:      array in which events will be logged
    mesh:             the system mesh 
    dset:             the system setup dataset 
    case:             The checkpoint case index
    """
    
    beta    = util.calc_beta(fields)
    w       = util.calc_lorentz_gamma(fields)
    t_prime = dset['time'] * time_scale
    gamma_min, gamma_max = args.gamma_lims 
    #================================================================
    #                    HYDRO CONDITIONS
    #================================================================
    p     = 2.5 # electron particle number index   
    eps_b = 0.1
    eps_e = 0.1
    # number of random angular samples if not fully 3D 
    angular_samples = 10
    r               = mesh['r']
    ndim = 1 
    if 'theta' in mesh:
        nim += 1
        theta = mesh['theta']
    else:
        theta = np.linspace(0.0, np.pi, angular_samples)
        
    if 'phi' in mesh:
        ndim += 1
        phi  = mesh['phi']
    else:
        phi = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    
    thetta, phii, rr = np.meshgrid(theta, phi, r)
    
    rhat          = np.array([np.sin(thetta)*np.cos(phii), np.sin(thetta)*np.sin(phii), np.cos(thetta)])  # radiail diirectional unit vector

    rho_einternal = fields['p'] * pre_scale / (dset['ad_gamma'] - 1.0)                                    # internal energy density
    bfield        = calc_bfield_shock_other(rho_einternal, eps_b)                                            # magnetic field based on equipartition
    ub            = bfield**2 / (8.0 * np.pi)                                                             # magnetic energy density
    n_e           = fields['rho'] * w * rho_scale / const.m_p.cgs                                         # electron number density
        
    # Each cell will have its own photons distribution. 
    # To account for this, we divide the gamma bins up 
    # and bin the photons in each cell with respect to the 
    # gamma bin
    dgamma     = (gamma_max - gamma_min) / 100.0
    gamma_bins = np.arange(gamma_min, gamma_max, dgamma)
    n_photons  = np.zeros(shape=(gamma_bins.size, mesh['r'].size))
    photon_erg = n_photons.copy()
    for idx, gamma_e in enumerate(gamma_bins):
        gamma_sample   = gen_random_from_powerlaw(gamma_e, gamma_e + dgamma, -p)
        nu_c           = gamma_sample ** 2 * nu_g
        nphot          = calc_nphotons_per_bin(dvolume, n_e, nu_g, gamma_sample, beta, ub, dt) * dgamma
        photon_erg[idx]= const.h.cgs * nu_c * nphot
        n_photons[idx] = nphot
    
    n_photons  = n_photons.T  
    photon_erg = photon_erg.T
    
    #================================================================
    #                 DIRECTIONAL SAMPLING RULES
    #================================================================
    rng = np.random.default_rng()
    
    # ============================================================
    # Source Trajectory
    phi_prime     = 2.0 * np.pi * rng.uniform(0, 1, size=phii.shape)
    mu_prime      = 2.0 * rng.uniform(0, 1, size=thetta.shape) - 1.0
    rhat          = np.array([np.sin(thetta)*np.cos(phii), np.sin(thetta)*np.sin(phii), np.cos(thetta)])
    nhat_prime    = np.array(
                        [np.sin(np.arccos(mu_prime)) * np.cos(phi_prime),
                         np.sin(np.arccos(mu_prime)) * np.sin(phi_prime),
                         mu_prime])
    
    # Cosine of the isotropic emission angle wrt to the propagation direction
    mu_zeta         = vector_dotproduct(rhat, nhat_prime)
    nvec_prime      = (r * length_scale) * nhat_prime 
    nprime_para     = vector_magnitude(nvec_prime) * mu_zeta * rhat  # Direction of emission parallel to propagation direction
    nprime_par_hat  = nprime_para / vector_magnitude(nprime_para)    # unit vecor for parallel direction 
    nprime_perp     = nvec_prime - nprime_para                       # Direction of emission perpendicular to propagation direction
    nprime_perp_hat = nprime_perp / vector_magnitude(nprime_perp)    # unit vector for parallel direction 
    beta_vec        = beta * rhat                                    # 3D-ify the flow velocity into cartesian 
    
    # Begin transorming the source trajectory into the rest frame
    beta_full       = (beta_vec[0]**2 + beta_vec[1]**2 + beta_vec[2]**2)**0.5          # Create pseduomesh for full flow velocity
    cos_ang_rest    = (mu_zeta + beta_full)/(1 + beta_full*mu_zeta)                    # cos of the resulting beamed angle in the plane of rhat and nhat prime
    rot_angle       = np.pi / 2.0 - np.arccos(cos_ang_rest)                            # rotation angle from initial emission direction to beaming direction
    
    # Resultant propagation direction in the lab frame
    nvec_rest     = ( (vector_magnitude(nprime_para) * np.cos(rot_angle) + vector_magnitude(nprime_perp) * np.sin(rot_angle)) * nprime_par_hat 
                    + ( - vector_magnitude(nprime_para) * np.sin(rot_angle) + vector_magnitude(nprime_perp) * np.cos(rot_angle)) * nprime_perp_hat 
                      ) / (1.0 - cos_ang_rest)**0.5
    nvec_mag      = vector_magnitude(nvec_rest)
    nhat_rest     = nvec_rest / nvec_mag

    photon_grid               = np.zeros(shape = (*rr.shape, len(gamma_bins)))
    four_momentum_prime       = photon_grid.copy()
    if ndim == 1:
        photon_grid[:, :]         = n_photons
        four_momentum_prime[:, :] = photon_erg
    elif ndim == 2:
        photon_grid[:] = n_photons
        four_momentum_prime[:] = photon_erg
    else:
        photon_grid         = n_photons
        four_momentum_prime = photon_erg
        
    # Some checks to make sure boosting direction was correct
    radial_cell   = beta.argmax()
    iso_angle   = np.arccos(mu_zeta)[0][0][radial_cell]
    beam_angle  = np.arccos(cos_ang_rest[0][0][radial_cell])
    beam_angle2 = np.arcsin(vector_dotproduct(nhat_rest, nhat_prime))[0][0][radial_cell]
    
    d = (1.0 * units.Mpc).to(units.cm)              # distance to observer
    t = t_prime - rr * length_scale / const.c.cgs   # total time to the observer
    
    # Source energy to rest frame 4-momentum
    four_momentum_rest = four_momentum_prime.copy()
    for idx, v in enumerate(beta):
        four_momentum_rest[:, :, idx] *= w[idx] * (1.0 + v)
    
    print(f"1/Gamma: {np.rad2deg(np.arcsin(1/w[radial_cell])) * units.deg:.>43.2f}")
    print(f'beta: {beta[radial_cell]:.>46.2f}')
    print(f'isotropic emission angle: {np.rad2deg(iso_angle) * units.deg:.>27.2f}')
    print(f'angle after beaming: {np.rad2deg(beam_angle) * units.deg:.>32.2f}')
    print(f'angle b/w rhat and nhat: {np.rad2deg(beam_angle2):.>28.2f}')
    print(f'time in observer frame: {t[0][0][radial_cell]:.>31.2f}')
    
def main():
    parser = argparse.ArgumentParser(description='Synchrotron Radiaition  Module')
    parser.add_argument('files', nargs='+', help='Explicit filenames or directory')
    parser.add_argument('--units', '-u', default=False, action='store_true', dest='units')
    parser.add_argument('--sradius', dest='srad', help='Fraction in solar radii of stellar radius', default=0.65, type=float)
    parser.add_argument('--dt', dest='dt', help='time step (in seconds) between checkpoints', default=0.1, type=float)
    parser.add_argument('--nu', dest='nu', help='Observed frequency', default=1e9, type=float, nargs='+')
    parser.add_argument('--gamma_lims', dest='gamma_lims', help='lorentz gamma limits for electron distro', default=[1.0,100.0], nargs='+', type=float)
    parser.add_argument('--2d', help='Set if files are 2d checkpts', dest='files2d', default=False, action='store_true')
    parser.add_argument('--3d', help='Set if files are 3d checkpts', dest='files3d', default=False, action='store_true')
    
    args = parser.parse_args()
    #check if path is a file
    isFile = os.path.isfile(args.files[0])

    #check if path is a directory
    isDirectory = os.path.isdir(args.files[0])
    
    if isDirectory:
        file_path = os.path.join(args.files[0], '')
        files = sorted([file_path + f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))])
    else:
        files = args.files 
    
    events_list = np.zeros(shape=(len(files), 2))
    flux_dict = {i: [] for i in range(len(args.nu))}
    past           = {}
    past['tobs']   = 0
    for nidx, nu in enumerate(args.nu):
        discrete_times = []
        for idx, file in enumerate(files):
            if args.files2d:
                fields, setup, mesh = util.read_2d_file(args, file)
                sari_piran_narayan_99(fields, args, frequency=nu, discrete_times=discrete_times, flux_array = flux_dict[nidx], mesh=mesh, dset=setup, case=idx)
            else:
                fields, setup, mesh = util.read_1d_file(file)
                sari_piran_narayan_99(fields, args, frequency=nu, discrete_times=discrete_times, flux_array = flux_dict[nidx], mesh=mesh, dset=setup, past=past, case=idx)
        
    total_time = (discrete_times * units.s).to(units.day)
    nbins      = 20
    time_bins  = np.geomspace(total_time[total_time > 0].min(), total_time.max(), nbins)
    time_bins  = np.insert(time_bins, 0, 0.0)
    
    discrete_times = np.asanyarray(discrete_times)
    for key, val in flux_dict.items():
        flux_dict[key] = np.asanyarray(val)
    
    fig, ax = plt.subplots(figsize=(8,8))
    freqs = np.array(args.nu) * units.Hz
    for key in flux_dict.keys():
        flux_per_tbin = np.zeros_like(time_bins.value)
        for idx, t in enumerate(time_bins):
            flux_per_tbin[idx] = flux_dict[key][total_time < t].sum()
            flux_dict[key][total_time < t] = 0.0
        ax.plot(time_bins, flux_per_tbin, label=rf'$\nu={freqs[key]:.2e}$')
    
    ax.set_title('Light curve for spherical BMK Test')
    ax.set_xlim(time_bins.value.min(), time_bins.value.max())
    ax.set_yscale('log')
    ax.set_xscale('symlog')
    ax.set_xlabel(r'$t_{\rm obs} [\rm day]$')
    ax.set_ylabel(r'$\rm Flux \ [\mu \rm Jy]$')
    ax.legend()
    fig.savefig('bmk_lightcurve_test.pdf')
    plt.show()
    
if __name__ == '__main__':
    main()