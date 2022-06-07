#! /usr/bin/env python


import numpy as np 
import astropy.constants as const 
import rad_hydro
import astropy.units as units 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import utility as util 
import argparse
import os 
import cycler
import sys
import h5py 

try:
    import cmasher as cmr 
except ImportError:
    print("cannot find cmasher module, using basic matplotlib colors insteads")

# FONT SIZES
SMALL_SIZE   = 8
DEFAULT_SIZE = 10
BIGGER_SIZE  = 12


    
def sari_piran_narayan_99(
    fields:         dict, 
    args:           argparse.ArgumentParser, 
    tbin_edges:     np.ndarray,
    fbin_edges:     np.ndarray,
    flux_array:     np.ndarray,
    mesh:           dict, 
    dset:           dict, 
    storage:        dict,
    overplot:       bool=False, 
    subplot:        bool=False, 
    ax:             plt.Axes=None, 
    case:           int=0
) -> None:
    
    beta       = util.calc_beta(fields)
    w          = util.calc_lorentz_gamma(fields)
    t_prime    = dset['time'] * util.scales.time_scale
    t_emitter  = t_prime / w
    #================================================================
    #                    HYDRO CONDITIONS
    #================================================================
    p     = 2.5  # Electron number index
    eps_b = 0.1  # Magnetic field fraction of internal energy 
    eps_e = 0.1  # shocked electrons fraction of internal energy
    
    rho_einternal = fields['p'] * util.scales.pre_scale / (dset['ad_gamma'] - 1.0)        # internal energy density
    bfield        = util.calc_bfield_shock(rho_einternal, eps_b)                          # magnetic field based on equipartition
    n_e_proper    = fields['rho'] * util.scales.rho_scale / const.m_p.cgs                 # electron number density
    nu_g          = util.calc_gyration_frequency(bfield)                                  # gyration frequency
    d             = 1e28 * units.cm                                                       # distance to source
    gamma_min     = util.calc_minimum_lorentz(eps_e, rho_einternal, n_e_proper, p)        # Minimum Lorentz factor of electrons 
    gamma_crit    = util.calc_critical_lorentz(bfield, t_emitter)                         # Critical Lorentz factor of electrons

    # step size between checkpoints
    if 'dt' in dset:
        dt  = dset['dt'] * util.scales.time_scale
    else:
        dt  = args.dt * util.scales.time_scale                                      

    # no moving mesh yet, so this is fine
    if case == 0:
        on_axis = False
        # on axis observers don't need phi zones
        util.generate_pseudo_mesh(args, mesh, expand_space=True)
        
        # Calc cell volumes
        if on_axis:
            dvolume = util.calc_cell_volume2D(mesh['rr'], mesh['thetta']) * util.scales.length_scale **3
        else:
            dvolume = util.calc_cell_volume3D(mesh['rr'], mesh['thetta'], mesh['phii']) * util.scales.length_scale ** 3
        
        if on_axis:
            rhat = np.array([np.sin(mesh['thetta']), np.zeros_like(mesh['thetta']), np.cos(mesh['thetta'])])  # radial unit vector        
        else:
            rhat = np.array([np.sin(mesh['thetta'])*np.cos(mesh['phii']), np.sin(mesh['thetta'])*np.sin(mesh['phii']), np.cos(mesh['thetta'])])  # radial unit vector  
        
        # Place observer along chosen axis
        theta_obs  = np.deg2rad(args.theta_obs) * np.ones_like(mesh['thetta'])
        obs_hat    = np.array([np.sin(theta_obs), np.zeros_like(mesh['thetta']), np.cos(theta_obs)])
        if on_axis:
            obs_idx    = util.find_nearest(mesh['thetta'][:,0], np.deg2rad(args.theta_obs))[0]
        else:
            obs_idx    = util.find_nearest(mesh['thetta'][0,:,0], np.deg2rad(args.theta_obs))[0]
            
        storage['obs_idx'] = obs_idx
        # Store everything in a dictionary that is constant
        storage['rhat']     = rhat 
        storage['obs_hat']  = obs_hat 
        storage['ndim']     = ndim
        storage['dvolume']  = dvolume
        storage['on_axis']  = on_axis
        storage['rr']       = mesh['rr']
        t_obs   = t_prime - rr * util.scales.length_scale * util.vector_dotproduct(rhat, obs_hat) / const.c.cgs
    else:
        dt_chkpt = t_prime - storage['t_prime']
        t_obs    = storage['t_obs'] + dt_chkpt

    beta_vec = beta * storage['rhat']
    obs_hat  = storage['obs_hat']
    
    # Calculate the maximum flux based on the average bolometric power per electron
    nu_c              = util.calc_nu(gamma_crit, nu_g)                                   # Critical frequency
    nu_m              = util.calc_nu(gamma_min, nu_g)                                    # Minimum frequency
    delta_doppler     = util.calc_doppler_delta(w, beta_vector=beta_vec, n_hat=obs_hat)  # Doppler factor
    emissivity        = util.calc_emissivity(bfield, n_e_proper, p)                      # Emissibity per cell 
    total_power       = storage['dvolume'] * emissivity                                  # Total emitted power per unit frequency in each cell volume
    flux_max          = total_power * delta_doppler ** 2.0                               # Maximum flux 
    
    storage['t_obs']   = t_obs
    storage['t_prime'] = t_prime
    t_obs              = t_obs.to(units.day)
    
    # the effective lifetime of the emitting cell must be accounted for
    dt_obs = tbin_edges[1:] - tbin_edges[:-1]
    dt_day = dt.to(units.day)

    # loop through the given frequencies and put them in their respective locations in dictionary
    for freq in args.nu:
        # The frequency we see is doppler boosted, so account for that
        nu_source = freq * units.Hz / delta_doppler
        power_cool = util.calc_powerlaw_flux(mesh, flux_max, p, nu_source, nu_c, nu_m, ndim = storage['ndim'], on_axis = storage['on_axis'])
        ff = (power_cool / (4.0 * np.pi * d **2)).to(units.mJy)
        
        # place the fluxes in the appropriate time bins
        t_obs_day = t_obs.to(units.day)
        for idx, t1 in enumerate(tbin_edges[:-1]):
            t2 = tbin_edges[idx + 1]
            trat = dt_day / dt_obs[idx] if case != 0 else 1.0
            flux_array[freq][idx] += trat * ff[(t_obs > t1) & (t_obs < t2)].sum()
            print(f"victory!")
            print(f"{t1} --- {t2}")
            print(f"bin:{t2 - t1}")
            print(f"t_prime: {t_prime}")
            print(f"observer time: {t_obs_day[0][0]}")
            print(f"added flux: {trat * ff[0].sum()}");
            print(f"boosted power: {flux_max[0,0].value}")
            print(f"power cool: {power_cool[0,0].value}")
            print(f"before conversion: {(power_cool / (4.0 * np.pi * d **2))[0,0]}")
            zzz = input('')
            
    # nu_obs       = util.calc_nu(w, nu_g=nu_g)
    # tdays        = [1]
    # for tday in tdays:
    #     for idx, nu1 in enumerate(fbin_edges[:-1]):
    #         nu2       = fbin_edges[idx + 1]        
    #         nu_source = np.sqrt(nu1 * nu2) / delta_doppler
    #         ff        = util.calc_powerlaw_flux(mesh, flux_max, p, nu_source, nu_c, nu_m, ndim = ndim, on_axis = storage['on_axis'])
    #         ff        = (ff / (4.0 * np.pi * d **2)).to(units.mJy)
            
    #         flux_array[tday][idx] += dt_day / dt_obs[idx] * ff[(nu_obs > nu1) & (nu_obs < nu2)].sum()
        
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
    t_prime = dset['time'] * util.scales.time_scale
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

    rho_einternal = fields['p'] * util.scales.pre_scale / (dset['ad_gamma'] - 1.0)                                    # internal energy density
    bfield        = calc_bfield_shock_other(rho_einternal, eps_b)                                         # magnetic field based on equipartition
    ub            = bfield**2 / (8.0 * np.pi)                                                             # magnetic energy density
    n_e           = fields['rho'] * w * util.scales.rho_scale / const.m_p.cgs                                         # electron number density
        
    # Each cell will have its own photons distribution. 
    # To account for this, we divide the gamma bins up 
    # and bin the photons in each cell with respect to the 
    # gamma bin
    dgamma     = (gamma_max - gamma_min) / 100.0
    gamma_bins = np.arange(gamma_min, gamma_max, dgamma)
    n_photons  = np.zeros(shape=(gamma_bins.size, mesh['r'].size))
    photon_erg = n_photons.copy()
    for idx, gamma_e in enumerate(gamma_bins):
        gamma_sample   = util.gen_random_from_powerlaw(gamma_e, gamma_e + dgamma, -p)
        nu_c           = gamma_sample ** 2 * nu_g
        nphot          = util.calc_nphotons_per_bin(dvolume, n_e, nu_g, gamma_sample, beta, ub, dt) * dgamma
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
    mu_zeta         = util.vector_dotproduct(rhat, nhat_prime)
    nvec_prime      = (r * util.scales.length_scale) * nhat_prime 
    nprime_para     = util.vector_magnitude(nvec_prime) * mu_zeta * rhat  # Direction of emission parallel to propagation direction
    nprime_par_hat  = nprime_para / util.vector_magnitude(nprime_para)    # unit vecor for parallel direction 
    nprime_perp     = nvec_prime - nprime_para                       # Direction of emission perpendicular to propagation direction
    nprime_perp_hat = nprime_perp / util.vector_magnitude(nprime_perp)    # unit vector for parallel direction 
    beta_vec        = beta * rhat                                    # 3D-ify the flow velocity into cartesian 
    
    # Begin transorming the source trajectory into the rest frame
    beta_full       = (beta_vec[0]**2 + beta_vec[1]**2 + beta_vec[2]**2)**0.5          # Create pseduomesh for full flow velocity
    cos_ang_rest    = (mu_zeta + beta_full)/(1 + beta_full*mu_zeta)                    # cos of the resulting beamed angle in the plane of rhat and nhat prime
    rot_angle       = np.pi / 2.0 - np.arccos(cos_ang_rest)                            # rotation angle from initial emission direction to beaming direction
    
    # Resultant propagation direction in the lab frame
    nvec_rest     = ( (util.vector_magnitude(nprime_para) * np.cos(rot_angle) + util.vector_magnitude(nprime_perp) * np.sin(rot_angle)) * nprime_par_hat 
                    + ( - util.vector_magnitude(nprime_para) * np.sin(rot_angle) + util.vector_magnitude(nprime_perp) * np.cos(rot_angle)) * nprime_perp_hat 
                      ) / (1.0 - cos_ang_rest)**0.5
    nvec_mag      = util.vector_magnitude(nvec_rest)
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
    t = t_prime - rr * util.scales.length_scale / const.c.cgs   # total time to the observer
    
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
    parser.add_argument('--theta_samples', dest='theta_samples', type=int, help='number of theta_samples', default=None)
    parser.add_argument('--phi_samples', dest='phi_samples', type=int, help='number of phi', default=10)
    parser.add_argument('--example_curve', dest='example_curve', type=str, help='data file from example light curves', default=None)
    parser.add_argument('--data_files', dest='data_files', type=str, help='data file from self computed light curves', default=None, nargs='+')
    parser.add_argument('--cmap', help='colormap scheme for light curves', dest='cmap', default=None, type=str)
    parser.add_argument('--clims', help='color value limits', dest='clims', nargs='+', type=float, default=[0.25, 0.75])
    parser.add_argument('--file_save', dest='file_save', help='name of file to be saved as', type=str, default='some_lc.h5')
    parser.add_argument('--example_label', dest='example_label', help='label of the example curve\'s markers', type=str, default='example')
    parser.add_argument('--xlims', dest='xlims', help='x limits in plot', default=None, type=float, nargs='+')
    parser.add_argument('--ylims', dest='ylims', help='y limits in plot', default=None, type=float, nargs='+') 
    parser.add_argument('--fig_dims', dest='fig_dims', help='figure dimensions', default=(5,4), type=float, nargs='+')
    parser.add_argument('--title', dest='title', help='title of plot', default=None)
    try:
        parser.add_argument('--compute', dest='compute', 
                            help='turn off if you have a data file you just want to plot immediately', 
                            action=argparse.BooleanOptionalAction, default=True)
    except:
        parser.add_argument('--compute', dest='compute', 
                            help='turn off if you have a data file you just want to plot immediately', 
                            action='store_false', default=True)
        
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
        files = sorted([file for file in args.files])
    
    # sort by length of strings now
    files.sort(key=len, reverse=False)
    
    if args.dim == 2:
        file_reader = util.read_2d_file 
    else:
        file_reader = util.read_1d_file
    
    fig_dims      = args.fig_dims 
    freqs         = np.array(args.nu) * units.Hz
    fig, ax       = plt.subplots(figsize=args.fig_dims)
    nbins         = args.ntbins
    nbin_edges    = nbins + 1
    tbin_edge     = util.get_tbin_edges(args, file_reader, files)
    tbin_edges    = np.geomspace(tbin_edge[0]*0.9, tbin_edge[1]*1.1, nbin_edges)
    time_bins     = np.sqrt(tbin_edges[1:] * tbin_edges[:-1])
    flux_per_bin = {i: np.zeros(nbins) * units.mJy for i in args.nu}
    events_list   = np.zeros(shape=(len(files), 2))
    storage       = {}
    
    # Make freq bins for spectra\
    tday             = np.array([1, 10, 100, 1000])
    fbin_edges       = np.geomspace(8e8, 3e17, nbin_edges) * units.Hz
    freq_bins        = np.sqrt(fbin_edges[1:] * fbin_edges[:-1])
    # flux_per_bin.update({j: np.zeros(nbins) * units.mJy for j in tday})
    
    if args.cmap is not None:
        vmin, vmax = args.clims 
        cinterval = np.linspace(vmin, vmax, len(args.nu))
        cmap      = plt.cm.get_cmap(args.cmap)
        colors    = util.get_colors(cinterval, cmap, vmin, vmax)
    else:
        colors     = ['c', 'y', 'm', 'k'] # list of basic colors
        
    linestyles = ['-','--','-.',':']  # list of basic linestyles
    lines = ["-","--","-.",":"]
    linecycler = cycler.cycle(lines)
    
    scales_dict = {
        'time_scale':   util.scales.time_scale.value,
        'length_scale': util.scales.length_scale.value,
        'rho_scale':    util.scales.rho_scale.value,
        'pre_scale':    util.scales.pre_scale.value,
        'v_scale':      1.0
    }   
    theta_obs = np.deg2rad(args.theta_obs)
    sim_info = {
        'theta_obs':       theta_obs,
        'nus':             freqs.value
    }
    
    
    if args.compute:
        for idx, file in enumerate(files):
            fields, setup, mesh = file_reader(file)
            # Generate a pseudo mesh if computing off-axis afterglows
            util.generate_pseudo_mesh(args, mesh)
            
            sim_info['dt']              = setup['dt']
            sim_info['adiabatic_gamma'] = setup['ad_gamma']
            sim_info['current_time']    = setup['time']
            rad_hydro.py_calc_fnu_2d(
                fields     = fields, 
                tbin_edges = tbin_edges.value,
                fbin_edges = fbin_edges.value,
                flux_array = flux_per_bin,
                mesh       = mesh, 
                qscales    = scales_dict, 
                sim_info   = sim_info,
                chkpt_idx  = idx,
                data_dim   = args.dim
            )
            
            # sari_piran_narayan_99(fields, args, tbin_edges=tbin_edges, fbin_edges=freq_bins_edges, flux_array = flux_per_bin, mesh=mesh, dset=setup, storage=storage, case=idx)
            # zzz = input('')
            print(f"Processed file {file}", flush=True)
    
    
    sim_lines = [0] * len(args.nu)
    for nidx, freq in enumerate(args.nu):
        power_of_ten = int(np.floor(np.log10(freq)))
        front_part   = freq / 10**power_of_ten 
        if front_part == 1.0:
            freq_label = r'10^{%d}'%(power_of_ten)
        else:
            freq_label = r'%f \times 10^{%fid}'%(front_part, power_of_ten)

        color = colors[nidx % len(args.nu)]
        sim_lines[nidx], = ax.plot(time_bins, flux_per_bin[freq], color=color, label=r'$\nu={} \rm Hz$'.format(freq_label))
        
        if args.example_curve is not None:
            example_data = util.read_example_afterglow_data(args.example_curve)
            nu_unit = freq * units.Hz
            ax.plot(example_data['tday'], example_data['light_curve_pcj'][nu_unit], 'o', color=color, markersize=0.5)
        
        if args.data_files is not None:
            for dfile in args.data_files:
                dat = util.read_my_datafile(dfile)
                nu_unit = freq * units.Hz
                ax.plot(dat['tday'], dat['fnu'][nu_unit], color=color, markersize=0.5, linestyle=next(linecycler))

    # Save the data
    if args.compute:
        file_name = args.file_save
        if os.path.splitext(args.file_save)[1] != '.h5':
            file_name += '.h5'
        
        isFile = os.path.isfile(file_name)
        dirname = os.path.dirname(file_name)
        if os.path.exists(dirname) == False and dirname != '':
            if not isFile:
                # Create a new directory because it does not exist 
                os.makedirs(dirname)
                print(80*'=')
                print(f"creating new directory named {dirname}...")
            
        print(80*"=")
        print(f"Saving file as {file_name}...")
        print(80*'=')
        with h5py.File(file_name, 'w') as hf: 
            fnu_save = np.array([flux_per_bin[key] for key  in flux_per_bin.keys()])
            dset = hf.create_dataset('sogbo_data', dtype='f')
            hf.create_dataset('nu',   data=[nu for nu in args.nu])
            hf.create_dataset('fnu',  data=fnu_save)
            hf.create_dataset('tbins', data=time_bins)
    
    if args.xlims is not None:
        tbound1, tbound2 = np.asanyarray(args.xlims) * units.day 
    else:
        tbound1 = time_bins[0]
        tbound2 = time_bins[-1]
        
    if args.title is not None:
        if args.dim == 1:
            ax.set_title(r'$ \rm Light \  curve \ for \ spherical \ BMK \ Test$')
        else:
            ax.set_title(r'$ \rm Light \ curve \ for \ conical \ BMK \ Test$')
    
    ylims = args.ylims if args.ylims else (1e-11, 1e4)
    ax.set_xlim(tbound1.value, tbound2.value)
    ax.set_ylim(*ylims)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(r'$t_{\rm obs} [\rm day]$')
    ax.set_ylabel(r'$\rm Flux \ Density \ [\rm mJy]$')
    ax.legend()
    if args.example_curve is not None:
        label = args.example_label
        example = mlines.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor='grey', markersize=5)
        
        ax.legend(handles=[*sim_lines, example])
        ax.axvline(3.5, linestyle='--', color='red')
    if args.save:
        file_str = f"{args.save}".replace(' ', '_')
        print(f'saving as {file_str}')
        fig.savefig(f'{file_str}.pdf')
        plt.show()
    else:
        plt.show()
    
if __name__ == '__main__':
    main()