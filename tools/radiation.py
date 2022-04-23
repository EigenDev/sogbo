#! /usr/bin/env python


import numpy as np 
import astropy.constants as const 
import astropy.units as units 
import matplotlib.pyplot as plt 
import utility as util 
import argparse
import os 
import sys

from typing import Callable

# FONT SIZES
SMALL_SIZE   = 8
DEFAULT_SIZE = 10
BIGGER_SIZE  = 12

if 'hydro_scales' in sys.argv:
    #================================
    #   Hydro scales
    #================================
    R_0   = const.R_sun.cgs 
    c     = const.c.cgs
    m_sun = const.M_sun.cgs
    rho_scale     = m_sun / (4./3. * np.pi * R_0 ** 3) 
    e_scale       = m_sun * c **2
    pre_scale     = e_scale / (4./3. * np.pi * R_0**3)
    time_scale    = R_0 / c
    length_scale  = R_0
else:
    #==============
    # BMK Scales
    #==============
    e_scale      = 1e53 * units.erg 
    rho_scale    = 1.0 * const.m_p.cgs / units.cm**3 
    length_scale = ((e_scale / (rho_scale * const.c.cgs**2))**(1/3)).to(units.cm)
    time_scale   = length_scale / const.c.cgs 
    pre_scale    = e_scale / length_scale ** 3

def generate_mesh(args: argparse.ArgumentParser, mesh: dict):
    """
    Generate a real or pseudo 3D mesh based on checkpoint data
    
    Parameters
    --------------------------
    args: argparser arguments from CLI 
    mesh: the mesh data from the checkpoint
    
    Return
    --------------------------
    rr:      (np.ndarray) 3D array of radial points
    thetta:  (np.ndarray) 3D array of polar points
    phii:    (np.ndarray) 3D arrray of azimuthal points
    ndim:    (int)        number of dimension from checkpoint mesh
    """
    
    r = mesh['r']
    ndim = 1
    if 'theta' in mesh:
        ndim += 1
        theta = mesh['theta']
    else:
        dlogr         = np.log10(r.max()/r.min())/(r.size - 1)
        theta_max     = np.pi / 2 if not args.full_sphere else np.pi
        theta_samples = int(thets_max / dlogr + 1)
        theta         = np.linspace(0.0, theta_max, theta_samples)
        
    if 'phi' in mesh:
        ndim += 1
        phi  = mesh['phi']
    else:
        phi_samples = 10
        phi = np.linspace(0.0, 2.0 * np.pi, phi_samples)

    thetta, phii, rr = np.meshgrid(theta, phi, r)
    
    return rr, thetta, phii, ndim
# Define function for string formatting of scientific notation
def get_tbin_edges(
    args: argparse.ArgumentParser, 
    file_reader: Callable,
    files: str, 
    *file_args):
    """
    Get the bin edges of the lightcurves based on the checkpoints given
    
    Parameters:
    -----------------------------------
    files: list of files to draw data from
    
    Returns:
    -----------------------------------
    tmin, tmax: tuple of time bins in units of days
    """
    fields_init,  setup_init,  mesh_init  = file_reader(files[0], *file_args)
    fields_final, setup_final, mesh_final = file_reader(files[-1],*file_args)
    
    t_beg = setup_init['time'] * time_scale
    t_end = setup_final['time'] * time_scale
    
    rr0, thetta, phii = generate_mesh(args, mesh_init)[:-1]
    rrf               = generate_mesh(args, mesh_final)[0]
    rhat              = np.array([np.sin(thetta)*np.cos(phii), np.sin(thetta)*np.sin(phii), np.cos(thetta)])  # radial unit vector                                                                                 # electron particle number index   
    
    # Place observer along chosen axis
    theta_obs  = np.deg2rad(args.theta_obs) * np.ones_like(thetta)
    obs_hat    = np.array([np.sin(theta_obs)*np.cos(phii), np.sin(theta_obs) * np.sin(phii), np.cos(theta_obs)])

    t_obs_min  = t_beg - rr0 * length_scale * vector_dotproduct(rhat, obs_hat) / const.c.cgs
    t_obs_max  = t_end - rrf * length_scale * vector_dotproduct(rhat, obs_hat) / const.c.cgs

    tmin = (t_obs_min[t_obs_min > 0].min()).to(units.day)
    tmax = (t_obs_max[t_obs_max > 0].max()).to(units.day)
    
    return tmin, tmax

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

# def calc_bfield_shock(rho: float, lorentz_gamma: float, beta: float, eps_b: float = 0.1) -> float:
#     """
#     Calc magnitude of magnetic field assuming shock jump conditions
    
#     Params:
#     ------------------------------------------------
#     rho:           fluid density in rest frame 
#     lorentz_gamma: lorentz factor 
#     beta:          dimensionless velocity of flow 
#     eps_b:            magnetic energy fraction
    
#     Return:
#     ------------------------------------------------
#     The magnetic field magnitude 
#     """
#     return (8.0 * np.pi * eps_b * rho * rho_scale)**0.5 * lorentz_gamma * beta * const.c.cgs

def calc_bfield_shock(einternal: float, eps_b: float) -> float:
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
    
def calc_gyration_frequency(b_field: float) -> float:
    """
    return gyration frequency in units of MHz
    Params
    ---------------------
    b_field: magnetic field value 
    
    Return
    ---------------------
    the gyration frequency 
    """
    # frequency_for_unit_field = (const.e.gauss * 1.0 * units.gauss) / (2.0 * np.pi * const.m_e.cgs * const.c.cgs)
    # print(frequency_for_unit_field)
    frequency_for_unit_field = (3.0 / 16.0) * (const.e.gauss * 1.0 * units.gauss) / (const.m_e.cgs * const.c.cgs)
    # print(frequency_for_unit_field)
    # zzz = input('')
    # zzz = input('')
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

def calc_emissivity(bfield: float, n: float, p: float) -> float:
    """Calculate the peak emissivity per frequency per equation (1) in
    https://iopscience.iop.org/article/10.1088/0004-637X/722/1/235/pdf
    """ 
    return 0.88 * (16.0/3.0)**2 * (p - 1) / (3.0 * p - 1.0) * (const.m_e.cgs * const.c.cgs ** 2 * const.sigma_T.cgs) / (8.0 * np.pi * const.e.gauss) * n * bfield

def calc_gamma_min(eps_e: float,e_thermal: float, n: float, p: float) -> float:
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
    return eps_e * (p - 2.0) / (p - 1.0) * e_thermal / (n * const.m_e.cgs * const.c.cgs**2)

def calc_powerlaw_flux(
        mesh:     dict,
        flux_max: float, 
        p:        float,
        nu:       float, 
        nu_c:     float, 
        nu_m:     float, 
        ndim:     int = 1) -> float:
        """
        ---------------------------------------
        Compute the flux according to https://arxiv.org/abs/astro-ph/9712005
        ---------------------------------------
        """
        f_nu = flux_max.copy()
        slow_cool    = nu_c > nu_m
        fast_cool    = nu_c < nu_m
        
        fast_break1  = nu_c > nu 
        fast_break2  = nu   > nu_m
        fast_mask1   = fast_cool & fast_break1 
        fast_mask2   = fast_cool & fast_break2
        fast_mask3   = fast_cool & (fast_break1 == False) & (fast_break2 == False)
        
        slow_break1  = nu_m > nu 
        slow_break2  = nu   > nu_c
        slow_mask1   = slow_cool & slow_break1
        slow_mask2   = slow_cool & slow_break2
        slow_mask3   = slow_cool & (slow_break1 == False) & (slow_break2 == False)
        if ndim == 1:
            f_nu[:, :, slow_mask1] *= (nu / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[:, :, slow_mask2] *= (nu_c[slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))*(nu / nu_c[slow_mask2])**(-0.5 * p)
            f_nu[:, :, slow_mask3] *= (nu / nu_m[slow_mask3])**(-0.5 * (p - 1.0))
            
            f_nu[:, :, fast_mask1] *= (nu / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[:, :, fast_mask2] *= (nu_m[fast_mask2] / nu_c[fast_mask2])**(-0.5)*(nu / nu_m[fast_mask2])**(-0.5 * p)
            f_nu[:, :, fast_mask3] *= (nu / nu_c[fast_mask3])**(-0.5)
            
            # print("a")
            # r = mesh['r']
            # for ridx in range(r.size):
            #     nu_crit = nu_c[ridx]
            #     nu_min  = nu_m[ridx]
            #     cooling = 'slow' if nu_crit > nu_min else 'fast'
            #     if cooling == 'fast':
            #         if nu_crit > nu:
            #             f_nu[:, :, ridx]  *= (nu/nu_crit)**(1.0/3.0)
            #         elif nu > nu_min:
            #              f_nu[:, :, ridx] *= (nu_min/nu_crit)**(-0.5)*(nu/nu_min)**(-0.5 * p)
            #         else:
            #             f_nu[:, :, ridx]  *= (nu/nu_crit)**(-0.5) 
            #     else:
            #         if nu_min > nu:
            #             f_nu[:, :, ridx] *= (nu/nu_min)**(1.0/3.0)
            #         elif nu > nu_crit:
            #             f_nu[:, :, ridx] *= (nu_crit/nu_min)**(-0.5 * (p-1))*(nu/nu_crit)**(-0.5 * p) 
            #         else:
            #             f_nu[:, :, ridx] *= (nu/nu_min) **(-0.5*(p-1.0))
            # print(f_nu[0])
            # print("b")
            # zzz = input('')
        else:
            f_nu[:, slow_mask1] *= (nu / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[:, slow_mask2] *= (nu_c[slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))*(nu / nu_c[slow_mask2])**(-0.5 * p)
            f_nu[:, slow_mask3] *= (nu / nu_m[slow_mask3])**(-0.5 * (p - 1.0))
            
            f_nu[:, fast_mask1] *= (nu / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[:, fast_mask2] *= (nu_m[fast_mask2] / nu_c[fast_mask2])**(-0.5)*(nu / nu_m[fast_mask2])**(-0.5 * p)
            f_nu[:, fast_mask3] *= (nu / nu_c[fast_mask3])**(-0.5)
            # print("a")
            # theta        = mesh['theta']
            # r            = mesh['r']
            # for tidx in range(theta.size):
            #     for ridx in range(r.size):
            #         nu_crit = nu_c[tidx, ridx]
            #         nu_min  = nu_m[tidx, ridx]
            #         cooling = 'slow' if nu_crit > nu_min else 'fast'
            #         zone = tidx, ridx
            #         if cooling == 'fast':
            #             if nu_crit > nu:
            #                 f_nu[:, zone]  *= (nu/nu_crit)**(1.0/3.0)
            #             elif nu > nu_min:
            #                 f_nu[:, zone]  *= (nu_min/nu_crit)**(-0.5)*(nu/nu_min)**(-0.5 * p)
            #             else:
            #                 f_nu[:, zone]  *= (nu/nu_crit)**(-0.5) 
            #         else:
            #             if nu_min > nu:
            #                 f_nu[:, zone] *= (nu/nu_min)**(1.0/3.0)
            #             elif nu > nu_crit:
            #                 f_nu[:, zone] *= (nu_crit/nu_min)**(-0.5 * (p-1))*(nu/nu_crit)**(-0.5 * p) 
            #             else:
            #                 f_nu[:, zone] *= (nu/nu_min) **(-0.5*(p-1.0))
            # print(f_nu[0])
            # print("b")
            # zzz = input('')
        return f_nu 
    
def sari_piran_narayan_99(
    fields:         dict, 
    args:           argparse.ArgumentParser, 
    time_bins:      np.ndarray,
    flux_array:     np.ndarray,
    mesh:           dict, 
    dset:           dict, 
    storage:           dict,
    overplot:       bool=False, 
    subplot:        bool=False, 
    ax:             plt.Axes=None, 
    case:           int=0
) -> None:
    
    beta       = util.calc_beta(fields)
    w          = util.calc_lorentz_gamma(fields)
    t_prime    = dset['time'] * time_scale
    t_emitter  = t_prime / w 
    #================================================================
    #                    HYDRO CONDITIONS
    #================================================================
    p     = 2.5  # Electron number index
    eps_b = 0.1  # Magnetic field fraction of internal energy 
    eps_e = 0.1  # Electric field fraction of internal energy
    
    rho_einternal = fields['p'] * pre_scale / (dset['ad_gamma'] - 1.0)   # internal energy density
    bfield        = calc_bfield_shock(rho_einternal, eps_b)              # magnetic field based on equipartition
    ub            = bfield**2 / (8.0 * np.pi)                            # magnetic energy density
    n_e_proper    = fields['rho'] * rho_scale / const.m_p.cgs            # electron number density
    dt            = args.dt * time_scale                                 # step size between checkpoints
    nu_g          = calc_gyration_frequency(bfield)                      # gyration frequency
    
    gamma_min  = calc_gamma_min(eps_e, rho_einternal, n_e_proper, p)        # Minimum Lorentz factor of electrons 
    gamma_crit = calc_critical_lorentz(bfield, t_emitter)                # Critical Lorentz factor of electrons
    
    r = mesh['r']
    d = 1e28 * units.cm
    if case == 0:
        rr, thetta, phii, ndim = generate_mesh(args, mesh)

        # Calc cell volumes
        dvolume = util.calc_cell_volume3D(rr, thetta, phii) * length_scale ** 3
        rhat    = np.array([np.sin(thetta)*np.cos(phii), np.sin(thetta)*np.sin(phii), np.cos(thetta)])  # radiail diirectional unit vector                                                                                 # electron particle number index   
        
        # Place observer along chosen axis
        theta_obs  = np.deg2rad(args.theta_obs) * np.ones_like(thetta)
        obs_hat    = np.array([np.sin(theta_obs)*np.cos(phii), np.sin(theta_obs) * np.sin(phii), np.cos(theta_obs)])
        
        # store everything in a dictionary that is constant
        storage['rhat']    = rhat 
        storage['obs_hat'] = obs_hat 
        storage['ndim']    = ndim
        storage['dvolume'] = dvolume
        t_obs   = t_prime - rr * length_scale * vector_dotproduct(rhat, obs_hat) / const.c.cgs
    else:
        dt_chkpt = t_prime - storage['t_prime']
        t_obs    = storage['t_obs'] + dt_chkpt

    beta_vec = beta * storage['rhat']
    obs_hat  = storage['obs_hat']
    ndim     = storage['ndim']
    dvolume  = storage['dvolume']
    
    # Calculate the maximum flux based on the average bolometric power per electron
    alpha             = 0.5 * (p - 1.0)                                             # Spectral power law index (might not need after all)
    nu_c              = calc_nu(gamma_crit, nu_g)                                   # Critical frequency
    nu_m              = calc_nu(gamma_min, nu_g)                                    # Minimum frequency
    delta_doppler     = calc_doppler_delta(w, beta_vector=beta_vec, n_hat=obs_hat)  # Doppler factor
    emissivity        = calc_emissivity(bfield, n_e_proper, p)                      # Emissibity per cell 
    total_power       = storage['dvolume'] * emissivity                             # Total emitted power per unit frequency in each cell volume
    flux_max          = total_power * delta_doppler ** (2.0)                        # Maximum flux 
    
    storage['t_obs']     = t_obs
    storage['t_max']     = t_obs.max()
    storage['t_prime']   = t_prime
    dt_obs               = time_bins[1:] - time_bins[:-1]
    dt_day               = dt.to(units.day)
    t_obs                = t_obs.to(units.day)
    
    # loop through the given frequencies and put them in their respective locations in dictionary
    for freq in args.nu:
        ff = calc_powerlaw_flux(mesh, flux_max, p, freq * units.Hz, nu_c, nu_m, ndim = ndim)
        ff = (ff / (4.0 * np.pi * d **2)).to(units.Jy)
            
        # place the fluxes in the appropriate time bins
        for idx, t1 in enumerate(time_bins[:-1]):
            t2 = time_bins[idx + 1]
            flux_array[freq][idx] += dt_day / dt_obs[idx] * ff[(t_obs > t1) & (t_obs < t2)].sum()
        
def log_events(
    fields:        dict, 
    args:          argparse.ArgumentParser, 
    events_list:   list,
    mesh:          dict, 
    dset:          dict, 
    case:          int=0) -> None:
    """
    Log the photon events based on synchrotron radiation (no asborption)
    
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
    parser.add_argument('--t0', dest='t0', help='time since simulation start', default=0.0, type=float)
    parser.add_argument('--tend', dest='tend', help='simulation end time', default=1.0, type=float)
    parser.add_argument('--tbin_start', dest='t_bin_start', help='where to start the time bins in units of days', default=1e-3, type=float)
    parser.add_argument('--gamma0', dest='gamma0', help='Initial Lorentz factor', default=1.0, type=float)
    parser.add_argument('--dt', dest='dt', help='time step (in seconds) between checkpoints', default=0.1, type=float)
    parser.add_argument('--theta_obs', dest='theta_obs', help='observation angle in degrees', type=float, default=0.0)
    parser.add_argument('--nu', dest='nu', help='Observed frequency', default=1e9, type=float, nargs='+')
    parser.add_argument('--gamma_lims', dest='gamma_lims', help='lorentz gamma limits for electron distro', default=[1.0,100.0], nargs='+', type=float)
    parser.add_argument('--dim', dest='dim', help='number of dimensions in checkpoin data', default=1, choices=[1,2,3], type=int)
    parser.add_argument('--full_sphere', help='Set if want to account for radiaition over whole sphere. Default is half', default=False, action='store_true')
    parser.add_argument('--save', help='file name to save fig', dest='save', default=None, type=str)
    parser.add_argument('--tex', help='true if want latex rendering on figs', default=False, action='store_true')
    parser.add_argument('--ntbins', dest='ntbins', type=int, help='number of time bins', default=50)
    args = parser.parse_args()

    if args.tex:
        plt.rc('text', usetex=True)
        
    #check if path is a file
    isFile = os.path.isfile(args.files[0])

    #check if path is a directory
    isDirectory = os.path.isdir(args.files[0])
    
    if isDirectory:
        file_path = os.path.join(args.files[0], '')
        files = sorted([file_path + f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))])
    else:
        files = args.files 
    
    if args.dim == 2:
        file_reader = util.read_2d_file
        func_args0  = [args]
    else:
        file_reader = util.read_1d_file
        func_args0  = []
        
    freqs         = np.array(args.nu) * units.Hz
    fig, ax       = plt.subplots(figsize=(8,8))
    nbins         = args.ntbins
    nbin_edges    = nbins + 1
    tbin_edge     = get_tbin_edges(args, file_reader, files, *func_args0)
    time_bins     = np.geomspace(tbin_edge[0]*0.9, tbin_edge[1]*1.1, nbin_edges)
    flux_per_tbin = {i: np.zeros(nbins) * units.Jy for i in args.nu}
    events_list   = np.zeros(shape=(len(files), 2))
    storage       = {}
    
    linestyles = ['-','--','-.',':'] # list of basic linestyles
    for idx, file in enumerate(files):
        func_args = func_args0 + [file]
        fields, setup, mesh = file_reader(*func_args)
        sari_piran_narayan_99(fields, args, time_bins=time_bins, flux_array = flux_per_tbin, mesh=mesh, dset=setup, storage=storage, case=idx)
        print(f"Processed file {file}", flush=True)
    
    t0 = args.t0 * time_scale.to(units.day)
    # time_bins *= 10**np.log10((t0 + time_bins) / time_bins)
    for nidx, freq in enumerate(args.nu):
        power_of_ten = int(np.floor(np.log10(freq)))
        front_part   = freq / 10**power_of_ten 
        if front_part == 1.0:
            freq_label = r'10^{%d}'%(power_of_ten)
        else:
            freq_label = r'%f \times 10^{%fid}'%(front_part, power_of_ten)
    
        style = linestyles[nidx % len(args.nu)]
        ax.plot(time_bins[:-1], 1e-3 * flux_per_tbin[freq], linestyle=style, label=r'$\nu={} \rm Hz$'.format(freq_label))
    

    tbound1    = time_bins[0]
    tbound2    = time_bins[-3]
    if args.dim == 1:
        ax.set_title('Light curve for spherical BMK Test')
    else:
        ax.set_title('Light curve for conical BMK Test')
        
    ax.set_xlim(tbound1.value, tbound2.value)
    ax.set_ylim(1e-14, 1e4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$t_{\rm obs} [\rm day]$')
    ax.set_ylabel(r'$\rm Flux \ Density \ [\rm mJy]$')
    ax.legend()
    
    if args.save:
        file_str = f"{args.save}".replace(' ', '_')
        print(f'saving as {file_str}')
        fig.savefig(f'{file_str}.pdf')
        plt.show()
    else:
        plt.show()
    
if __name__ == '__main__':
    main()