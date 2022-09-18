#! /usr/bin/env python

# BMK Test Problem in 1D

import numpy as np 
import argparse
import cycler
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import astropy.constants as const 
import astropy.units as units 
import h5py 
import os 

c          = const.c.cgs 
e_scale    = 1e53 * units.erg 
rho_scale  = 1.0 * const.m_p.cgs / units.cm**3 

def calc_gamma_shock(l: float, t: float, k: float) -> float:
    """Calc shock lorentz factor"""
    return ((17.0 - 4.0 * k) / 8.0 / np.pi)**0.5 * (l / t)**(3.0 / 2.0)

def calc_fluid_gamma_max(l:float, t: float, k: float) -> float:
    """Calc maximum lorentz factor of fluid"""
    gamma_shock = calc_gamma_shock(l, t, k)
    return gamma_shock / 2.0**0.5
    
def calc_chi(gamma_shock:float, r: float, t: float, m: float) -> float:
    """Similarity variable as function of time"""
    return (1.0 + 2.0 * (m + 1)* gamma_shock**2)*(1.0 - r / t)

def calc_gamma_fluid(gamma_shock: float, chi: float) -> float:
    return gamma_shock / (2.0 * chi)**0.5

def calc_rho(gamma_shock: float, gamma_fluid: float, chi: float, rho0: float, k: float) -> float:
    return 2.0 * rho0 * gamma_shock ** 2 * chi ** (-(7.0 - 2.0 * k) /(4.0 - k)) / gamma_fluid

def calc_pressure(gamma_shock: float, chi: float, rho0: float, k: float) -> float:
    return (2.0 / 3.0) * rho0 * gamma_shock ** 2 * chi**(-(17.0 - 4.0*k) / (12.0 - 3.0*k))

def calc_shock_radius(gamma_shock: float, t: float, m: float):
    return t * (1.0 - (2.0*(m + 1.0) * gamma_shock**2)**(-1))

def find_nearest(arr: np.ndarray, val: float):
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
    
def main():
    parser = argparse.ArgumentParser(description="Analytic BMK Solution")
    parser.add_argument('--e0',     help='initial energy input',      dest='e0',    type=float, default=1.0)
    parser.add_argument('--rho0',   help='initial density of medium', dest='rho0',  type=float, default=1.0)
    parser.add_argument('--t0',     help='iniiial sim time',          dest='t0',    type=float, default=0.01)
    parser.add_argument('--nzones', help='number of radial zones',    dest='nzones', type=int, default=4096)
    parser.add_argument('--rmax',   help='max radius', dest='rmax', type=float, default=1.0)
    parser.add_argument('--var',    help='select the variable you want to plot', dest='var', default = 'gamma_beta', choices=['gamma_beta', 'rho', 'pressure'])
    parser.add_argument('--m',      help='BMK self similarity parameter', default=3, type=float, dest='bmk_m')
    parser.add_argument('--k',      help='Density gradient slope', default=0, type=float, dest='k')
    parser.add_argument('--rinit',  help='initial blast_wave radius', default=0.01, type=float, dest='rinit')
    parser.add_argument('--data_dir', help='Data directory', default='data/bmk_oned', type=str, dest='data_dir')
    parser.add_argument('--tinterval',      help='time intervals to plot', default=0.1, type=float, dest='tinterval')
    parser.add_argument('--bottom', help='bottom ylim for plot', dest='bottom', default=0.0, type=float)
    parser.add_argument('--save',   help='flag to save figure. takes name of figure as arg', dest='save', default=None, type=str)
    parser.add_argument('--plot', dest='plot', help='set if want to see plot', default=False)

    args = parser.parse_args()
    
    data_dir   = args.data_dir
    dir_exists = os.path.exists(data_dir)
    
    if not dir_exists:
        os.makedirs(data_dir)
    
    data_dir = os.path.join(data_dir, '')
    if args.save is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": 10
            }
        )
        
    # Initial Conditions 
    e0     = args.e0           # initial energy
    rho0   = args.rho0         # initial density
    ell    = (e0/(rho0 * args.rinit**(-args.k)))**(1/(3 - args.k))  # inital length scale
    t      = args.t0           # initial simulation time
    
    gamma_shock_crit = 2.0 
    tphysical        = (((17.0 - 4.0 * args.k) / (8*np.pi)) * ell * gamma_shock_crit ** (-2.0))**(1.0/3.0)
    gamma_shock0     = calc_gamma_shock(ell, t, args.k)
    r0               = calc_shock_radius(gamma_shock0, t, args.bmk_m)
    nr               = args.nzones
    times            = np.geomspace(t, tphysical, nr)
    gamma_shock      = calc_gamma_shock(ell, times, args.k)
    r                = calc_shock_radius(gamma_shock, times, args.bmk_m)
    
    # Initial arrays
    gamma_fluid = np.ones_like(r)
    rho         = np.ones_like(r) * rho0 * (r/r0)**(-args.k)
    pressure    = rho * 1e-10
    
    gamma_fluid[0] = gamma_shock0 / (2.0**0.5)
    interval = 0.0
        
    if args.plot:
        fig, ax  = plt.subplots(1, 1, figsize=(4,4))
    
    i = 0
    t_last = 0
    
    for tidx, t in enumerate(times):  
        # Solution only physical when gamma_shock**2/2 >= chi
        chi_critical = 0.5 * gamma_shock[tidx]**2 
        chi          = calc_chi(gamma_shock[tidx], r, t, args.bmk_m)
        
        smask               = (chi >= 1.0)
        gamma_fluid[smask]  = calc_gamma_fluid(gamma_shock[smask], chi[smask])
        rho[smask]          = calc_rho(gamma_shock[smask], gamma_fluid[smask], chi[smask], rho0, args.k)
        pressure[smask]     = calc_pressure(gamma_shock[smask], chi[smask], rho0, args.k)
        
        gamma_fluid[gamma_fluid < 1]    = 1
        rho[chi > chi_critical]         = 1e-10 
        pressure[chi > chi_critical]    = 1e-10
        gamma_fluid[chi > chi_critical] = 1

        n_zeros = str(int(4 - int(np.floor(np.log10(i))) if i > 0 else 3))
        file_name = f'{data_dir}{args.nzones}.chkpt.{i:03}.h5'
        with h5py.File(f'{file_name}', 'w') as f:
            print(f'[Writing to {file_name}]')
            beta = (1.0 - gamma_fluid**(-2.0))**0.5
            sim_info = f.create_dataset('sim_info', dtype='i')
            f.create_dataset('rho', data=rho)
            f.create_dataset('p', data=pressure)
            f.create_dataset('v', data=beta)
            f.create_dataset('radii', data=r)
            sim_info.attrs['current_time'] = t 
            sim_info.attrs['dt']           = t - t_last
            sim_info.attrs['ad_gamma']     = 4.0 / 3.0 
            sim_info.attrs['x1min']        = r[0]
            sim_info.attrs['x1max']        = r[-1] 
            sim_info.attrs['Nx']           = args.nzones 
            sim_info.attrs['linspace']     = False 
            
        if args.plot:
            if args.var == 'gamma_beta':
                gb  = (gamma_fluid**2 - 1.0)**0.5
                ax.semilogx(r, gb)
                # ax.axvline(t, linestyle='--')
            elif args.var == 'rho':
                ax.semilogx(r, rho)
            elif args.var == 'pressure':
                ax.semilogx(r, pressure)
        
        t_last = t
        i += 1
    
    if args.plot:
        if args.var == 'gamma_beta':
            # Compare the t^-3/2 scaling with what was calculated
            ells  = np.asanyarray(ells)
            gamma_fluid_scaling = gamma_shock / 2.0 ** 0.5
            gamma_fluid_scaling[gamma_fluid_scaling < 1.0] = 1.0
            gb_scaling  = (gamma_fluid_scaling**2 - 1.0)**0.5
            ax.semilogx(times, gb_scaling, linestyle='--', label=r'$\Gamma \propto t^{-3/2}$')
            ax.legend()
        
        if args.var == 'rho':
            ylabel = r'$\rho$'
        elif args.var == 'pressure':
            ylabel = 'p'
        else:
            ylabel = r'$\gamma \beta_{\rm fluid}$'
        
        ax.set_title(f'1D BMK Problem at t = {t:.1f}, N={args.nzones}, k={args.k:.1f}')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$r$')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_xlim(r[0]*0.99, r[-1])
        ax.set_ylim(bottom=args.bottom)
        
        if not args.save:
            plt.show()
        else:
            fig.savefig("{}.pdf".format(args.save).replace(' ', '_'), dpi=600, bbox_inches='tight')
        
if __name__ == "__main__":
    main()