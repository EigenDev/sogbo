import argparse
import numpy as np 
import astropy.units as units 
cimport numpy as np 
cimport rad_hydro


def py_calc_fnu_2d(
    fields:         dict, 
    tbin_edges:     np.ndarray,
    fbin_edges:     np.ndarray,
    flux_array:     dict,
    mesh:           dict, 
    qscales:        dict, 
    sim_info:       dict,
    chkpt_idx:      int,
    data_dim:       int = 2
):
    flattened_fields = np.array(
        [fields['rho'].flatten(),
        fields['gamma_beta'].flatten(),
        fields['p'].flatten()], dtype=float
    )

    if 'phi' in mesh:
        flattened_mesh = np.asanyarray(
            [mesh['r'].flatten(),
            mesh['theta'].flatten(),
            mesh['phi']], dtype=object
        )
    else:
        flattened_mesh = np.asanyarray(
            [mesh['r'].flatten(),
            mesh['theta'].flatten()], dtype=object
        )

    stripped_flux = np.array(
        [flux_array[key].value.flatten() for key in flux_array.keys()], dtype=float
    )
    flattened_flux = stripped_flux.flatten()

    cdef sim_conditions sim_cond 
    cdef quant_scales quant_scales

    # Set the sim conditions
    sim_cond.dt           = sim_info['dt']
    sim_cond.current_time = sim_info['current_time']
    sim_cond.theta_obs    = sim_info['theta_obs']
    sim_cond.ad_gamma     = sim_info['adiabatic_gamma']
    sim_cond.nus          = sim_info['nus']

    # set the dimensional scales
    quant_scales.time_scale    = qscales['time_scale']
    quant_scales.pre_scale     = qscales['pre_scale']
    quant_scales.rho_scale     = qscales['rho_scale']
    quant_scales.v_scale       = 1.0 
    quant_scales.length_scale  = qscales['length_scale']

    calculated_flux = calc_fnu_2d(
                        sim_cond,
                        quant_scales,
                        flattened_fields, 
                        flattened_mesh,
                        tbin_edges,
                        fbin_edges,
                        flattened_flux, 
                        chkpt_idx,
                        data_dim
                    )

    py_fluxes = np.array(calculated_flux).reshape(len(flux_array.keys()), len(tbin_edges[:-1]))
    for idx, key in enumerate(flux_array.keys()):
        flux_array[key] = py_fluxes[idx] * units.mJy