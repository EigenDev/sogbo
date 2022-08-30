# distutils: language = c++

from libcpp.vector cimport vector
cdef extern from "rad_units.hpp" namespace "sogbo_rad":
    cdef struct sim_conditions:
        double dt, theta_obs, ad_gamma, current_time
        vector[double] nus

    cdef struct quant_scales:
        double time_scale, pre_scale, rho_scale, v_scale, length_scale

    cdef vector[double] calc_fnu_2d(
        sim_conditions args,
        quant_scales  qscales,
        vector[vector[double]] fields, 
        vector[vector[double]] mesh,
        vector[double] tbin_edges,
        vector[double] flux_array, 
        int chkpt_idx,
        int data_dim
    )