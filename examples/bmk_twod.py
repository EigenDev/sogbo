#! /usr/bin/env python

# BMK Test Problem in 2D

import numpy as np 
import argparse
import h5py
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import os 

def calc_shock_lorentz_gamma(l: float, t: float, k: float) -> float:
    """Calc shock lorentz factor"""
    return ((17.0 - 4.0 * k) / 8.0 / np.pi)**0.5 * (l / t)**(3.0 / 2.0)

def calc_fluid_gamma_max(l:float, t: float, k: float) -> float:
    """Calc maximum lorentz factor of fluid"""
    gamma_shock = calc_shock_lorentz_gamma(l, t, k)
    return gamma_shock / 2.0**0.5
    
def calc_chi(l:float, r: float, t: float, m: float, k: float) -> float:
    """Similarity variable as function of time"""
    gamma_shock = calc_shock_lorentz_gamma(l, t, k)
    return (1.0 + 2.0 * (m + 1)* gamma_shock**2)*(1.0 - r / t)

def calc_gamma_fluid(gamma_shock: float, chi: float) -> float:
    return gamma_shock / (2.0 * chi)**0.5

def calc_rho(gamma_shock: float, chi: float, rho0: float, k: float) -> float:
    gamma_fluid = calc_gamma_fluid(gamma_shock, chi)
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
    parser.add_argument('--tend',   help='dimensionless time to end simulation',    type=float, dest='tend', default=0.8)
    parser.add_argument('--nr',     help='number of radial zones',    dest='nr', type=int, default=128)
    parser.add_argument('--rmax',   help='max radius', dest='rmax', type=float, default=1.0)
    parser.add_argument('--var',    help='select the variable you want to plot', dest='var', default = 'gamma_beta', choices=['gamma_beta', 'rho', 'pressure'])
    parser.add_argument('--m',      help='BMK self similarity parameter', default=3, type=float, dest='bmk_m')
    parser.add_argument('--k',      help='Density gradient slope', default=0, type=float, dest='k')
    parser.add_argument('--rinit',  help='initial blast_wave radius', default=0.01, type=float, dest='rinit')
    parser.add_argument('--data_dir', help='Data directory', default='data/bmk_twod', type=str, dest='data_dir')
    parser.add_argument('--tinterval',      help='time intervals to plot', default=0.1, type=float, dest='tinterval')
    parser.add_argument('--theta_j',help='Opening angle of blast wave cone', default=np.pi, type=float, dest='theta_j')
    parser.add_argument('--tidx',   help='index for viewing angle of blast wave', default=0, type=int, dest='tidx')
    parser.add_argument('--nd_plot',   help='set if want full 2D plot', default=False, action='store_true', dest='nd_plot')
    parser.add_argument('--full_sphere',   help='set if want to account for full sphere', default=False, action='store_true', dest='full_sphere')
    parser.add_argument('--save',   help='flag to save figure. takes name of figure as arg', dest='save', default=None, type=str)
    parser.add_argument('--plot', dest='plot', help='set if want to see plot', default=False, action='store_true')
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
    ell    = (e0/(rho0 * args.rinit**args.k))**(1/(3 - args.k))  # inital length scale
    t      = args.t0           # initial simulation time
    
    tphysical     = ((17.0 - 4.0 * args.k) / (8*np.pi))**(1/3) * ell * 2.0 ** (-1.0/3.0)
    gamma_shock0  = calc_shock_lorentz_gamma(ell, t, args.k)
    r0            = calc_shock_radius(gamma_shock0, t, args.bmk_m)
    # grid constraints
    theta_max     = np.pi
    theta_min     = 0.0
    nr            = args.nr 
    dlogr         = np.log10(tphysical / r0) / (nr - 1)
    npolar        = int(theta_max / dlogr + 1)
    theta         = np.linspace(theta_min, theta_max, npolar)
    times         = np.geomspace(t, tphysical, nr)
    gamma_shock   = calc_shock_lorentz_gamma(ell, times, args.k)
    r             = calc_shock_radius(gamma_shock, times, args.bmk_m)
    rr, thetta    = np.meshgrid(r, theta)
    
    # Initial arrays
    theta_j_idx  = find_nearest(theta, args.theta_j)[0]
    gamma_fluid = np.ones_like(rr)
    rho         = np.ones_like(rr) * rho0 * (r/r0)**(-args.k)
    pressure    = rho * 1e-10
    
    gamma_fluid[:theta_j_idx, 0] = gamma_shock0 / (2.0**0.5)
    gamma_fluid[theta.size - theta_j_idx:, 0] = gamma_shock0 / (2.0**0.5)
    interval = 0.0
    if args.plot:
        if not args.nd_plot:
            fig, ax  = plt.subplots(1, 1, figsize=(4,4))
    
    i = 0
    t_last = 0
    for tidx, t in enumerate(times):
        # Solution only physical when gamma_shock**2/2 >= chi
        chi_critical = 0.5 * gamma_shock[tidx]**2
        chi          = calc_chi(ell, r, t, args.bmk_m, args.k)
        
        smask        = (chi >= 1.0)
        # Northern jet
        rho[:theta_j_idx,   smask]       = calc_rho(gamma_shock[smask], chi[smask], rho0, args.k)
        gamma_fluid[:theta_j_idx, smask] = calc_gamma_fluid(gamma_shock[smask], chi[smask])
        pressure[:theta_j_idx, smask]    = calc_pressure(gamma_shock[smask], chi[smask], rho0, args.k)
        
        # Southern jet
        if theta_max == np.pi:
            rho[theta.size   - theta_j_idx:, smask]       = calc_rho(gamma_shock[smask], chi[smask], rho0, args.k)
            gamma_fluid[theta.size - theta_j_idx:, smask] = calc_gamma_fluid(gamma_shock[smask], chi[smask])
            pressure[theta.size - theta_j_idx:, smask]    = calc_pressure(gamma_shock[smask], chi[smask], rho0, args.k)
        
        gamma_fluid[gamma_fluid < 1]       = 1
        rho[:, chi > chi_critical]         = 1e-10 
        pressure[:, chi > chi_critical]    = 1e-10
        gamma_fluid[:, chi > chi_critical] = 1
        if True:
            n_zeros = str(int(4 - int(np.floor(np.log10(i))) if i > 0 else 3))
            file_name = f'{data_dir}{npolar}.chkpt.{i:03}.h5'
            with h5py.File(f'{file_name}', 'w') as f:
                print(f'[Writing to {file_name}]')
                beta = (1.0 - (gamma_fluid)**(-2.0))**0.5
                beta1 = beta 
                beta2 = 0 * beta 
                sim_info = f.create_dataset('sim_info', dtype='i')
                f.create_dataset('rho',   data=rho)
                f.create_dataset('p',     data=pressure)
                f.create_dataset('v1',    data=beta1)
                f.create_dataset('v2',    data=beta2)
                f.create_dataset('radii', data=r)
                sim_info.attrs['current_time'] = t 
                sim_info.attrs['dt']           = t - t_last
                sim_info.attrs['ad_gamma']     = 4.0 / 3.0 
                sim_info.attrs['x1min']        = r0 
                sim_info.attrs['x1max']        = args.rmax 
                sim_info.attrs['x2min']        = theta_min
                sim_info.attrs['x2max']        = theta_max
                sim_info.attrs['nx']           = nr 
                sim_info.attrs['ny']           = npolar
                sim_info.attrs['linspace']     = False 
            
            if args.plot:
                if not args.nd_plot:
                    if args.var == 'gamma_beta':
                        gb  = (gamma_fluid**2 - 1.0)**0.5
                        ax.semilogx(r, gb[args.tidx])
                    elif args.var == 'rho':
                        ax.semilogx(r, rho[args.tidx])
                    elif args.var == 'pressure':
                        ax.semilogx(r, pressure[args.tidx])
            
            # print(t - t_last)
            # zzz = input('')
            t_last = t 
            i += 1

    if args.plot:
        if not args.nd_plot:
            if args.var == 'gamma_beta':
                # Compare the t^-3/2 scaling with what was calculated
                ells  = np.asanyarray(ells)
                gamma_shock_scaling = gamma_shock / 2.0**0.5
                gamma_shock_scaling[gamma_shock_scaling < 1.0] = 1.0
                gb_scaling  = (gamma_shock_scaling**2 - 1.0)**0.5
                ax.semilogx(times, gb_scaling, linestyle='--', label=r'$\Gamma \propto t^{-3/2}$')
                ax.legend()
            
            if args.var == 'rho':
                ylabel = r'$\rho$'
            elif args.var == 'pressure':
                ylabel = 'p'
            else:
                ylabel = r'$\gamma \beta_{\rm fluid}$'
            
            ax.set_title(rf'2D BMK Problem at t = {t:.1f}, $\theta$ ={theta[args.tidx]:.1f} N = {npolar} $\times$ {nr}, k={args.k:.1f}')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(r'$r/\ell$')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlim(r0*0.99, args.rmax)
            ax.set_ylim(bottom=0.0)
        else:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            norm = mcolors.LogNorm(vmin=None, vmax=None)
            if args.var == 'rho':
                c = ax.pcolormesh(thetta, rr, rho, norm=norm, shading='auto')
                ax.pcolormesh(-thetta, rr, rho, norm=norm, shading='auto')
                ylabel = r'$\rho$'
            elif args.var == 'pressure':
                c = ax.pcolormesh(thetta, rr, pressure, norm=norm, shading='auto')
                ax.pcolormesh(-thetta, rr, pressure, norm=norm, shading='auto')
                ylabel = 'p'
            else:
                norm = mcolors.PowerNorm(gamma=0.5)
                c = ax.pcolormesh(thetta, rr, gamma_fluid, norm=norm, shading='auto')
                ax.pcolormesh(-thetta, rr, gamma_fluid, norm=norm, shading='auto')
                ylabel = r'$\gamma \beta_{\rm fluid}$'
            
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # ax.set_theta_direction(-1)
            cbax = fig.colorbar(c, orientation='vertical')
            cbax.set_label(ylabel)
            
        if not args.save:
            plt.show()
        else:
            fig.savefig("{}.pdf".format(args.save).replace(' ', '_'), dpi=600, bbox_inches='tight')
        
if __name__ == "__main__":
    main()