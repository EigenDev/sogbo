#! /usr/bin/env python 

# Utility functions for visualization scripts 

import h5py 
import astropy.constants as const 
import astropy.units as units
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
    
from typing import Union
from typing import Callable

c = const.c.cgs
# FONT SIZES
class Scale:
    def __init__(self, scale='blandford_mckee'):
        if scale == 'hydro':
            #================================
            #   Hydro scales
            #================================
            self.length_scale  = const.R_sun.cgs 
            self.m_sun         = const.M_sun.cgs
            self.rho_scale     = self.m_sun / (4./3. * np.pi * self.length_scale ** 3) 
            self.e_scale       = self.m_sun * c **2
            self.pre_scale     = self.e_scale / (4./3. * np.pi * self.length_scale**3)
            self.time_scale    = self.length_scale / c
        elif scale == 'blandford_mckee':
            #==============
            # BMK Scales
            #==============
            self.e_scale      = 1e53 * units.erg 
            self.rho_scale    = 1.0 * const.m_p.cgs / units.cm**3 
            self.length_scale = ((self.e_scale / (self.rho_scale * const.c.cgs**2))**(1/3)).to(units.cm)
            self.time_scale   = self.length_scale / const.c.cgs 
            self.pre_scale    = self.e_scale / self.length_scale ** 3  

scales = Scale()

def calc_rverticies(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return rvertices 

def calc_theta_verticies(theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    return tvertices 

def calc_solid_angle3D(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[:, 1:] + theta[:, :-1])
    tvertices = np.insert(tvertices, 0, theta[:, 0], axis=1)
    tvertices = np.insert(tvertices, tvertices.shape[1], theta[:, -1], axis=1)
    tcenter   = 0.5 * (tvertices[:,1:] + tvertices[:,:-1])
    dth       = tvertices[:,1: ] - tvertices[:, :-1]
    
    phi_vertices = 0.5 * (phi[1:] + phi[:-1])
    phi_vertices = np.insert(phi_vertices,  0, phi[0], axis=0)
    phi_vertices = np.insert(phi_vertices, phi_vertices.shape[0], phi[-1], axis=0)
    dphi         = phi_vertices[1: ] - phi_vertices[:-1]
    
    return np.sin(tcenter) * dth * dphi
    
def calc_cell_volume1D(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return 4.0 * np.pi * (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3)

def calc_cell_volume2D(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    dcos      = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    rvertices = np.sqrt(r[:, 1:] * r[:, :-1])
    rvertices = np.insert(rvertices,  0, r[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], r[:, -1], axis=1)
    dr        = rvertices[:, 1:] - rvertices[:, :-1]
    # print(f"{dr[0,1219] * scales.length_scale:.2e}")
    return 2.0 * np.pi *  (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos

def calc_cell_volume3D(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    pvertices = 0.5 * (phi[1:] + phi[:-1])
    pvertices = np.insert(pvertices, 0, phi[0], axis=0)
    pvertices = np.insert(pvertices, pvertices.shape[0], phi[-1], axis=0)
    dphi      = pvertices[1:] - pvertices[:-1]
    
    tvertices = 0.5 * (theta[:,1:] + theta[:,:-1])
    tvertices = np.insert(tvertices, 0, theta[:,0], axis=1)
    tvertices = np.insert(tvertices, tvertices.shape[1], theta[:,-1], axis=1)
    dcos      = np.cos(tvertices[:,:-1]) - np.cos(tvertices[:,1:])
    
    rvertices = np.sqrt(r[:,:, 1:] * r[:,:, :-1])
    rvertices = np.insert(rvertices,  0, r[:,:, 0], axis=2)
    rvertices = np.insert(rvertices, rvertices.shape[2], r[:,:, -1], axis=2)
    dr        = rvertices[:, :, 1:] - rvertices[:, :, :-1]
    
    dV = (1.0/3.0) * (rvertices[:,:, 1:]**3 - rvertices[:,:, :-1]**3) * dphi * dcos
    return dV

def calc_enthalpy(fields: dict) -> np.ndarray:
    return 1.0 + fields['p']*fields['ad_gamma'] / (fields['rho'] * (fields['ad_gamma'] - 1.0))
    
def calc_lorentz_gamma(fields: dict) -> np.ndarray:
    """
    The Lorentz factor of the full fiel 
    """
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict) -> np.ndarray:
    """
    Calculate speed of body with respect to speed of light 
    """
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def get_field_str(args: argparse.ArgumentParser) -> str:
    """
    Convert the args.field variable into a human readable string. 
    Supports Greek letters 
    """
    field_str_list = []
    for field in args.field:
        if field == 'rho' or field == 'D':
            var = r'\rho' if field == 'rho' else 'D'
            if args.units:
                field_str_list.append( r'${}$ [g cm$^{{-3}}$]'.format(var))
            else:
                field_str_list.append( r'${}/{}_0$'.format(var,var))
            
        elif field == 'gamma_beta':
            field_str_list.append( r'$\Gamma \beta$')
        elif field == 'gamma_beta_1':
            field_str_list.append( r'$\Gamma \beta_1$')
        elif field == 'gamma_beta_2':
            field_str_list.append( r'$\Gamma \beta_2$')
        elif field == 'energy' or field == 'p':
            if args.units:
                if field == 'energy':
                    field_str_list.append( r'$\tau [\rm erg \ cm^{{-3}}]$')
                else:
                    field_str_list.append( r'$p [\rm erg \ cm^{{-3}}]$')
            else:
                if field == 'energy':
                    field_str_list.append( r'$\tau/\tau_0$')
                else:
                    field_str_list.append( r'$p/p_0$')
        elif field == 'energy_rst':
            if args.units:
                field_str_list.append( r'$\tau + D \  [\rm erg \ cm^{-3}]$')
            else:
                field_str_list.append( r'$\tau + D')
        elif field == 'chi':
            field_str_list.append( r'$\chi$')
        elif field == 'chi_dens':
            field_str_list.append( r'$D \cdot \chi$')
        elif field == 'temperature':
            field_str_list.append("T [eV]" if args.units else "T")
        else:
            field_str_list.append(field)

    return field_str_list if len(args.field) > 1 else field_str_list[0]

def calc_lorentz_gamma(fields: dict) -> np.ndarray:
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def read_2d_file(filename: str) -> Union[dict,dict,dict]:
    """
    Read in hydro data from 2D checkpoint file 
    """
    setup  = {}
    fields = {}
    is_cartesian = False
    with h5py.File(filename, 'r') as hf: 
        ds          = hf.get('sim_info')
        rho         = hf.get('rho')[:]
        v1          = hf.get('v1')[:]
        v2          = hf.get('v2')[:]
        p           = hf.get('p')[:]
        t           = ds.attrs['current_time']
        
        try:
            x1max = ds.attrs['x1max']
            x1min = ds.attrs['x1min']
            x2max = ds.attrs['x2max']
            x2min = ds.attrs['x2min']
        except:
            x1max = ds.attrs['xmax']
            x1min = ds.attrs['xmin']
            x2max = ds.attrs['ymax']
            x2min = ds.attrs['ymin']  
        
        # New checkpoint files, so check if new attributes were
        # implemented or not
        try:
            nx          = ds.attrs['nx']
            ny          = ds.attrs['ny']
        except:
            nx          = ds.attrs['NX']
            ny          = ds.attrs['NY']
        
        try:
            chi = hf.get('chi')[:]
        except:
            chi = np.zeros((ny, nx))
            
        try:
            gamma = ds.attrs['adiabatic_gamma']
        except:
            gamma = 4./3.
        
        try:
            dt = ds.attrs['dt']
        except:
            pass
        
        # Check for garbage value
        if gamma < 1:
            gamma = 4./3. 
            
        try:
            coord_sysem = ds.attrs['geometry'].decode('utf-8')
        except Exception as e:
            coord_sysem = 'spherical'
            
        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
        
        setup['x1max'] = x1max 
        setup['x1min'] = x1min 
        setup['x2max'] = x2max 
        setup['x2min'] = x2min 
        setup['time']  = t
        
        rho = rho.reshape(ny, nx)
        v1  = v1.reshape(ny, nx)
        v2  = v2.reshape(ny, nx)
        p   = p.reshape(ny, nx)
        chi = chi.reshape(ny, nx)
        
        try:
            radii = hf.get('radii')[:]
        except:
            pass 
        
        if 'radii' not in locals():
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            chi = chi[2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        else:
            xactive = nx
            yactive = ny
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        
        
        if is_linspace:
            setup['x1'] = np.linspace(x1min, x1max, xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
        else:
            setup['x1'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
            setup['x2'] = np.linspace(x2min, x2max, yactive)
        
        if 'radii' in locals():
            setup['x1'] = radii 
        else:
            if is_linspace:
                setup['x1'] = np.linspace(x1min, x1max, xactive)
            else:
                setup['x1'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
        if coord_sysem == 'cartesian':
            is_cartesian = True
        
        # if (v1**2 + v2**2).any() >= 1.0:
        #     W = 0
        # else:
        #     W = 1/np.sqrt(1.0 -(v1**2 + v2**2))
        W = 1/np.sqrt(1.0 -(v1**2 + v2**2))
        beta = np.sqrt(v1**2 + v2**2)
        
        if 'dt' in locals():
            setup['dt']      = dt 
            
        fields['rho']          = rho
        fields['v1']           = v1 
        fields['v2']           = v2 
        fields['p']            = p
        fields['chi']          = chi
        fields['gamma_beta']   = W*beta
        fields['ad_gamma']     = gamma
        setup['ad_gamma']      = gamma
        setup['is_cartesian']  = is_cartesian
        
        
    ynpts, xnpts = rho.shape 
    mesh = {}
    if setup['is_cartesian']:
        xx, yy = np.meshgrid(setup['x1'], setup['x2'])
        mesh['xx'] = xx
        mesh['yy'] = yy
    else:      
        rr, tt = np.meshgrid(setup['x1'], setup['x2'])
        mesh['thetta'] = tt 
        mesh['rr']     = rr
        mesh['r']      = setup['x1']
        mesh['theta']  = setup['x2']
        
    return fields, setup, mesh 

def read_1d_file(filename: str) -> dict:
    """
    Read in the hydro data from 1D checkpoint 
    """
    is_linspace = False
    ofield = {}
    setups = {}
    mesh   = {}
    with h5py.File(filename, 'r') as hf:
        ds = hf.get('sim_info')
        
        rho         = hf.get('rho')[:]
        v           = hf.get('v')[:]
        p           = hf.get('p')[:]
        nx          = ds.attrs['Nx']
        t           = ds.attrs['current_time']
        try:
            x1max = ds.attrs['x1max']
            x1min = ds.attrs['x1min']
        except:
            x1max = ds.attrs['xmax']
            x1min = ds.attrs['xmin']

        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
        
        
        try:
            radii = hf.get('radii')[:]
        except:
            pass 
        try:
            dt = ds.attrs['dt']
        except:
            pass
        
        # rho = rho[2:-2]
        # v   = v  [2:-2]
        # p   = p  [2:-2]
        # xactive = nx - 4
        xactive = nx
        W    = 1/np.sqrt(1 - v**2)
        
        a    = (4 * const.sigma_sb.cgs / c)
        k    = const.k_B.cgs
        T    = (3 * p * scales.pre_scale  / a)**(1./4.)
        T_eV = (k * T).to(units.eV)
        
        h = 1.0 + 4/3 * p / (rho * (4/3 - 1))
        
        if 'radii' in locals():
            mesh['r'] = radii 
        else:
            if is_linspace:
                mesh['r'] = np.linspace(x1min, x1max, xactive)
            else:
                mesh['r'] = np.logspace(np.log10(x1min), np.log10(x1max), xactive)
        
        if 'dt' in locals():
            setups['dt']      = dt 
            
        setups['ad_gamma']    = 4./3.
        setups['time']        = t
        setups['linspace']    = is_linspace
        ofield['ad_gamma']    = 4./3.
        ofield['rho']         = rho
        ofield['v']           = v
        ofield['p']           = p
        ofield['W']           = W
        ofield['enthalpy']    = h
        ofield['gamma_beta']  = W*v
        ofield['temperature'] = T_eV
        mesh['xlims']         = x1min, x1max
        
    return ofield, setups, mesh

def prims2var(fields: dict, var: str) -> np.ndarray:
    """
    Converts the primitives to the specified variable
    """
    h = calc_enthalpy(fields)
    W = calc_lorentz_gamma(fields)
    if var == 'D':
        # Lab frame density
        return fields['rho'] * W
    elif var == 'S':
        # Lab frame momentum density
        return fields['rho'] * W**2 * calc_enthalpy(fields) * fields['v']
    elif var == 'energy':
        # Energy minus rest mass energy
        return fields['rho']*h*W**2 - fields['p'] - fields['rho']*W
    elif var == 'energy_rst':
        # Total Energy
        return fields['rho']*h*W**2 - fields['p']
    elif var == 'temperature':
        a    = (4.0 * const.sigma_sb.cgs / c)
        T    = (3.0 * fields['p'] * scales.pre_scale  / a)**0.25
        T_eV = (const.k_B.cgs * T).to(units.eV)
        return T_eV.value
    elif var == 'gamma_beta':
        return W * fields['v']
    elif var == 'chi_dens':
        return fields['chi'] * fields['rho'] * W
    elif var == 'gamma_beta_1':
        return W * fields['v1']
    elif var == 'gamma_beta_2':
        return W * fields['v2']
    elif var =='sp_enthalpy':
        # Specific enthalpy
        return h - 1.0  

def read_example_afterglow_data(filename: str) -> dict:
    """
    Reads afterglow data from afterglow library (Zhang and MacFadyen 2009 or van Eerten et al. 2010)
    """
    with h5py.File(filename, "r") as hf:
        nu   = hf.get('nu')[:]   * units.Hz
        t    = hf.get('t')[:]    * units.s 
        fnu  = hf.get('fnu')[:]  * 1e26 * units.mJy 
        fnu2 = hf.get('fnu2')[:] * 1e26 * units.mJy 
        
    tday = t.to(units.day)
    data_dict = {}
    data_dict['tday'] = tday 
    data_dict['freq'] = nu 
    data_dict['light_curve'] = {nu_val: fnu[i, :] for i, nu_val in enumerate(nu)}
    data_dict['light_curve_pcj'] = {nu_val: fnu[i, :] + fnu2[i, :] for i, nu_val in enumerate(nu)}
    data_dict['spectra'] = {tday_val: fnu[:, i] for i, tday_val in enumerate(tday)}
    data_dict['spectra_pcj'] = {tday_val: fnu2[:, i] for i, tday_val in enumerate(tday)}
    
    return data_dict

def read_my_datafile(filename: str) -> dict:
    """
    Reads afterglow data from afterglow library (Zhang and MacFadyen 2009 or van Eerten et al. 2010)
    """
    with h5py.File(filename, "r") as hf:
        nu   = hf.get('nu')[:]    * units.Hz
        tday = hf.get('tbins')[:] * units.day
        fnu  = hf.get('fnu')[:]   * units.mJy 
        

    data_dict = {}
    data_dict['tday'] = tday 
    data_dict['freq'] = nu 
    data_dict['fnu'] = {nu_val: fnu[i, :] for i, nu_val in enumerate(nu)}
    # data_dict['light_curve_pcj'] = {nu_val: fnu[i, :] + fnu2[i, :] for i, nu_val in enumerate(nu)}
    # data_dict['spectra'] = {tday_val: fnu[:, i] for i, tday_val in enumerate(tday)}
    # data_dict['spectra_pcj'] = {tday_val: fnu2[:, i] for i, tday_val in enumerate(tday)}
    
    return data_dict
def get_colors(interval: np.ndarray, cmap: plt.cm, vmin: float = None, vmax: float = None):
    """
    Return array of rgba colors for a given matplotlib colormap
    
    Parameters
    -------------------------
    interval: interval range for colormarp min and max 
    cmap: the matplotlib colormap instnace
    vmin: minimum for colormap 
    vmax: maximum for colormap 
    
    Returns
    -------------------------
    arr: the colormap array generate by the user conditions
    """
    norm = plt.Normalize(vmin, vmax)
    return cmap(norm(interval))
    
def find_nearest(arr: list, val: float) -> int:
    """ Return nearest index to val in array"""
    arr = np.asanyarray(arr)
    if arr.ndim == 1:
        idx = np.argmin(np.abs(arr - val))
        return idx, arr[idx]
    else:
        x   = np.abs(arr - val)
        idx = np.where(x == x.min())
        return idx 
    

def generate_pseudo_mesh(args: argparse.ArgumentParser, mesh: dict, expand_space: bool = False):
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
        theta_samples = int(theta_max / dlogr + 1) if not args.theta_samples else args.theta_samples 
        theta         = np.linspace(0.0, theta_max, theta_samples)
        mesh['theta'] = theta 
        
    if 'phi' in mesh:
        ndim += 1
        phi  = mesh['phi']
    else:
        phi = np.linspace(0.0, 2.0 * np.pi, args.phi_samples)
        mesh['phi']   = phi 
        
    if expand_space:
        thetta, phii, rr = np.meshgrid(theta, phi, r)
        mesh['rr'] = rr 
        mesh['thetta'] = thetta 
        mesh['phii'] = phii 
    
    
def get_tbin_edges(
    args: argparse.ArgumentParser, 
    file_reader: Callable,
    files: str):
    """
    Get the bin edges of the lightcurves based on the checkpoints given
    
    Parameters:
    -----------------------------------
    files: list of files to draw data from
    
    Returns:
    -----------------------------------
    tmin, tmax: tuple of time bins in units of days
    """
    on_axis = False
    if args.theta_obs == 0:
        on_axis = True
    
    setup_init,  mesh_init  = file_reader(files[0])[1:]
    setup_final, mesh_final = file_reader(files[-1])[1:]
    
    t_beg = setup_init['time']  * scales.time_scale
    t_end = setup_final['time'] * scales.time_scale
    
    if on_axis:
        generate_pseudo_mesh(args, mesh_init , expand_space=True)
        generate_pseudo_mesh(args, mesh_final, expand_space=True)
        rhat        = np.array([np.sin(mesh_init['thetta']), np.zeros_like(mesh_init['thetta']), np.cos(mesh_init['thetta'])])  # radial unit vector  
    else:
        generate_pseudo_mesh(args, mesh_init , expand_space=True)
        generate_pseudo_mesh(args, mesh_final, expand_space=True)
        rhat        = np.array([np.sin(mesh_init['thetta'])*np.cos(mesh_init['phii']), 
                                np.zeros_like(mesh_init['thetta'])*np.sin(mesh_init['phii']),
                                np.cos(mesh_init['thetta'])])
    
    # Place observer along chosen axis
    theta_obs_rad = np.deg2rad(args.theta_obs)
    theta_obs     = theta_obs_rad * np.ones_like(mesh_init['thetta'])
    obs_hat       = np.array([np.sin(theta_obs), np.zeros_like(mesh_init['thetta']), np.cos(theta_obs)])
    r_dot_nhat    = vector_dotproduct(rhat, obs_hat)
    
        
    t_obs_min  = t_beg - mesh_init['rr'] * scales.time_scale * r_dot_nhat
    t_obs_max  = t_end - mesh_final['rr'] * scales.time_scale * r_dot_nhat

    if on_axis:
        theta_idx = find_nearest(mesh_init['thetta'][:,0], theta_obs_rad)[0]
    else:
        theta_idx  = find_nearest(mesh_init['thetta'][0,:,0], theta_obs_rad)[0]
    t_obs_beam = t_obs_min[0][theta_idx]
    t_obs_beam = t_obs_beam[t_obs_beam >= 0]
    
    tmin       = (t_obs_beam.min()).to(units.day)
    tmax       = (t_obs_max[t_obs_max > 0].max()).to(units.day)
    
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
    frequency_for_unit_field = (3.0 / 4.0 / np.pi) * (const.e.gauss * 1.0 * units.gauss) / (const.m_e.cgs * const.c.cgs)
    return frequency_for_unit_field.value  * b_field.value * units.Hz

def calc_total_synch_power(lorentz_gamma: float, ub: float, beta: float) -> float:
    """
    Calc bolometric synhrotron power
    
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
    """Calculate the synchrotron frequency as function of lorentz_factor"""
    return gamma_e ** 2 * nu_g 

def calc_critical_lorentz(bfield: float, time: float)-> float:
    """Calculate the critical Lorentz factor as function of time and magnetic field"""
    return (6.0 * np.pi * const.m_e.cgs * const.c.cgs) / (const.sigma_T.cgs * bfield ** 2 * time)

def calc_max_power_per_frequency(bfield: float) -> float:
    """Calculate the maximum power per frequency""" 
    return (const.m_e.cgs * const.c.cgs ** 2 * const.sigma_T.cgs) / (3.0 * const.e.gauss) * bfield

def calc_emissivity(bfield: float, n: float, p: float) -> float:
    """Calculate the peak emissivity per frequency per equation (A3) in
    https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    """ 
    eps_m = (9.6323/ 8.0 / np.pi) * (p - 1.0) / (3.0 * p - 1.0) * (3.0)**0.5 * const.e.gauss**3 / (const.m_e.cgs * const.c.cgs**2) * n * bfield
    return eps_m

def calc_minimum_lorentz(eps_e: float,e_thermal: float, n: float, p: float) -> float:
    """
    Calculate the minimum lorentz factor of electrons in the distribution
    
    Params:
    ------------------------------
    eps_e:              fraction of internal energy due to shocked electrons
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
    on_axis:  bool = True,
    ndim:     int = 1,
    r: float = 1) -> float:
    """
    ---------------------------------------
    Compute the flux according to https://arxiv.org/abs/astro-ph/9712005
    ---------------------------------------
    """
    f_nu = flux_max.copy()
    slow_cool    = nu_c > nu_m
    fast_cool    = nu_c < nu_m
    
    fast_break1  = nu < nu_c
    fast_break2  = (nu < nu_m) & (nu > nu_c)
    fast_mask1   = fast_cool & fast_break1 
    fast_mask2   = fast_cool & fast_break2
    fast_mask3   = fast_cool & (fast_break1 == False) & (fast_break2 == False)
    
    slow_break1  = nu < nu_m 
    slow_break2  = (nu < nu_c) & (nu > nu_m)
    slow_mask1   = slow_cool & slow_break1
    slow_mask2   = slow_cool & slow_break2
    slow_mask3   = slow_cool & (slow_break1 == False) & (slow_break2 == False)
    
    if ndim == 1:
        if on_axis:
            # Collapse the masks into their respective 1D symmetries
            slow_mask1 = slow_mask1[0]
            slow_mask2 = slow_mask2[0]
            slow_mask3 = slow_mask3[0]
            
            fast_mask1 = fast_mask1[0]
            fast_mask2 = fast_mask2[0]
            fast_mask3 = fast_mask3[0]
            
            f_nu[:, slow_mask1] *= (nu[:, slow_mask1] / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[:, slow_mask2] *= (nu[:, slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))
            f_nu[:, slow_mask3] *= (nu_c[slow_mask3]  / nu_m[slow_mask3])**(-0.5 * (p - 1.0)) * (nu[:, slow_mask3] / nu_c[slow_mask3])**(-0.5 * p)
            
            f_nu[:, fast_mask1] *= (nu[:, fast_mask1] / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[:, fast_mask2] *= (nu[:, fast_mask2] / nu_c[fast_mask2])**(-0.5)
            f_nu[:, fast_mask3] *= (nu_m[fast_mask3]  / nu_c[fast_mask3])**(-0.5) * (nu[:, fast_mask3] / nu_m[fast_mask3])**(-0.5 * p)
            
        else:
            # Collapse the masks into their respective 1D symmetries
            slow_mask1 = slow_mask1[0][0]
            slow_mask2 = slow_mask2[0][0]
            slow_mask3 = slow_mask3[0][0]
            
            fast_mask1 = fast_mask1[0][0]
            fast_mask2 = fast_mask2[0][0]
            fast_mask3 = fast_mask3[0][0]
            
            f_nu[:, :, slow_mask1] *= (nu[:, :, slow_mask1] / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[:, :, slow_mask2] *= (nu[:, :, slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))
            f_nu[:, :, slow_mask3] *= (nu_c[slow_mask3]     / nu_m[slow_mask3])**(-0.5 * (p - 1.0)) * (nu[:, :, slow_mask3] / nu_c[slow_mask3])**(-0.5 * p)
            
            
            f_nu[:, :, fast_mask1] *= (nu[:, :, fast_mask1] / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[:, :, fast_mask2] *= (nu[:, :, fast_mask2] / nu_c[fast_mask2])**(-0.5)
            f_nu[:, :, fast_mask3] *= (nu_m[fast_mask3]     / nu_c[fast_mask3])**(-0.5) * (nu[:, :, fast_mask3] / nu_m[fast_mask3])**(-0.5 * p)
    else:
        if on_axis:
            f_nu[slow_mask1] *= (nu[slow_mask1] / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[slow_mask2] *= (nu[slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))
            f_nu[slow_mask3] *= (nu_c[slow_mask3]  / nu_m[slow_mask3])**(-0.5 * (p - 1.0))*(nu[slow_mask3] / nu_c[slow_mask3])**(-0.5 * p)
            
            f_nu[fast_mask1] *= (nu[fast_mask1] / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[fast_mask2] *= (nu[fast_mask2] / nu_c[fast_mask2])**(-0.5)
            f_nu[fast_mask3] *= (nu_m[fast_mask3]  / nu_c[fast_mask3])**(-0.5)*(nu[fast_mask3] / nu_m[fast_mask3])**(-0.5 * p)
        else:
            # Collapse the masks into their respective 2D symmetries
            slow_mask1 = slow_mask1[0]
            slow_mask2 = slow_mask2[0]
            slow_mask3 = slow_mask3[0]
            
            fast_mask1 = fast_mask1[0]
            fast_mask2 = fast_mask2[0]
            fast_mask3 = fast_mask3[0]
            
            f_nu[:, slow_mask1] *= (nu[:, slow_mask1] / nu_m[slow_mask1])**(1.0 / 3.0)  
            f_nu[:, slow_mask2] *= (nu[:, slow_mask2] / nu_m[slow_mask2])**(-0.5 * (p - 1.0))
            f_nu[:, slow_mask3] *= (nu_c[slow_mask3]  / nu_m[slow_mask3])**(-0.5 * (p - 1.0))*(nu[:, slow_mask3] / nu_c[slow_mask3])**(-0.5 * p)
            
            
            f_nu[:, fast_mask1] *= (nu[:, fast_mask1] / nu_c[fast_mask1])**(1.0 / 3.0)
            f_nu[:, fast_mask2] *= (nu[:, fast_mask2] / nu_c[fast_mask2])**(-0.5)
            f_nu[:, fast_mask3] *= (nu_m[fast_mask3]  / nu_c[fast_mask3])**(-0.5)*(nu[:, fast_mask3] / nu_m[fast_mask3])**(-0.5 * p)
            
        
    return f_nu