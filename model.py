#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:30:03 2024

@author: kir312
"""



import pandas as pd
import numpy as np
from scipy.integrate import odeint



outfolder = 'output/'


# -----------------------------------------------------------------------------
# ODE model functions
# -----------------------------------------------------------------------------



def ode_model(compartments_inits_flat, t, params):
    
    # expand initial compartments
    # S, E0, Ec, Ed, Ecd, Y0, Z0, Yc, Zc, Yd, Zd, Ycd, Zcd = compartments_inits
    S, E0, Ec, Ed, Ecd, Y0, Z0, Yc, Zc, Yd, Zd, Ycd, Zcd = compartments_inits_flat.reshape((13,3))
    
    # expand parameters
    mu_entry, mu_exit,b, K, f_c, f_d, f_cd, nat_recovery_rate, care_seeking_rate, screening_rate, retreatment_prob_symptomatic, retreatment_rate_symptomatic, prop_symptomatic, doxypep_uptake, doxypep_rr = params_inits 

    
    # susceptible 
    dSdt = mu_entry*np.sum((S, Y0, Z0, Yc, Zc, Yd, Zd, Ycd, Zcd), axis=0) - mu_exit*S \
            - np.matmul(K, b*(Y0+Z0+f_c*(Yc+Zc)+f_d*(Yd+Zd)+f_cd*(Ycd+Zcd)))*S \
            + nat_recovery_rate*(Y0+Z0+Yc+Zc+Yd+Zd+Ycd+Zcd) \
            + care_seeking_rate*(Y0+Yd) \
            + screening_rate*(Z0+Zd) \
            + retreatment_prob_symptomatic*retreatment_rate_symptomatic*(Yc+Ycd) \
            + doxypep_uptake*(1-doxypep_rr)*(E0+Ec)
            
    # exposed
    dE0dt = + np.matmul(K,b*(Y0+Z0))*S \
            - E0 

    dEcdt = + np.matmul(K, b*f_c*(Yc+Zc))*S \
            - Ec 

    dEddt = + np.matmul(K,b*f_d*(Yd+Zd))*S \
            - Ed 

    dEcddt = np.matmul(K, b*f_cd*(Ycd+Zcd))*S \
            - Ecd 

    # infectious - symptomatic
    new_Y0 = prop_symptomatic*(1-doxypep_uptake+doxypep_uptake*doxypep_rr)*E0
    dY0dt = new_Y0 \
            - nat_recovery_rate*Y0 \
            - care_seeking_rate*Y0  \
            - mu_exit*Y0
    
    new_Yc = prop_symptomatic*(1-doxypep_uptake+doxypep_uptake*doxypep_rr)*Ec
    dYcdt = new_Yc \
            - nat_recovery_rate*Yc \
            - retreatment_prob_symptomatic*retreatment_rate_symptomatic*Yc \
             - mu_exit*Yc
             
    new_Yd = prop_symptomatic*Ed
    dYddt = new_Yd \
            - nat_recovery_rate*Yd \
            - care_seeking_rate*Yd \
            - mu_exit*Yd
    
    new_Ycd = prop_symptomatic*Ecd
    dYcddt = new_Ycd \
            - nat_recovery_rate*Ycd \
            - retreatment_prob_symptomatic*retreatment_rate_symptomatic*Ycd \
             - mu_exit*Ycd

    # infectious - asymptomatic
    new_Z0 = (1-prop_symptomatic)*(1-doxypep_uptake+doxypep_uptake*doxypep_rr)*E0
    dZ0dt =  new_Z0 \
            - nat_recovery_rate*Z0 \
            - screening_rate*Z0 \
            - mu_exit*Z0
    
    new_Zc = (1-prop_symptomatic)*(1-doxypep_uptake+doxypep_uptake*doxypep_rr)*Ec
    dZcdt = new_Zc \
            - nat_recovery_rate*Zc \
             - mu_exit*Zc
    
    new_Zd = (1-prop_symptomatic)*Ed 
    dZddt =  new_Zd \
            - nat_recovery_rate*Zd \
            - screening_rate*Zd \
             - mu_exit*Zd
    
    new_Zcd = (1-prop_symptomatic)*Ecd
    dZcddt =  new_Zcd \
            - nat_recovery_rate*Zcd \
             - mu_exit*Zcd
    
    # output compartment changes
    out = np.stack((dSdt, dE0dt, dEcdt, dEddt, dEcddt, dY0dt, dZ0dt, dYcdt, dZcdt, dYddt, dZddt, dYcddt, dZcddt))
    out_flat = out.flatten()
    
    # keep track of the new infections (for sampling)
    incidences = np.stack((new_Y0, new_Z0, new_Yc, new_Zc, new_Yd, new_Zd, new_Ycd, new_Zcd ))
    
    # keep track of individuals seeking care
    care_screen = np.stack((care_seeking_rate*Y0, # treatment 0
                             care_seeking_rate*Yc, # retreatment c
                             care_seeking_rate*Yd, # treatment d
                             care_seeking_rate*Ycd, # retreatment cd
                             screening_rate*Z0, # screen 0
                             screening_rate*Zc, # screen c
                             screening_rate*Zd, # screen d
                             screening_rate*Zcd # screen cd
                             ))
    
    return out_flat, incidences, care_screen




# wrapper function
def ode_wrapper(compartments_inits_flat, t, params):
    out_comp, out_inc, out_care = ode_model(compartments_inits_flat, t, params)
    return out_comp


# -----------------------------------------------------------------------------
# parameters and starting conditions
# -----------------------------------------------------------------------------

# time interval: days


# population size
N_total = 10**6
# order of groups: high, intermediate, low
N_groups = np.array([0.1*N_total, 0.6*N_total, 0.3*N_total])

# population entry and exit 
mu_entry = 1/(20*365)
mu_exit = mu_entry

# -- transmission
# transmission probability (vector, by group)
b = 0.5517084
# contact matrix (2d, 3x3 matrix)
pc_rate_h = 20*1.4076374/365
pc_rate_m = 5*1.4076374/365
pc_rate_l = 1*1.4076374/365
pc_rates = [pc_rate_h, pc_rate_m, pc_rate_l]
assortativity = 0.2544778
id_matrix = np.identity(3)

K = assortativity*(np.divide(pc_rates,N_groups) * id_matrix) \
    + (1-assortativity)*(np.outer(pc_rates, pc_rates))/np.sum(np.multiply(pc_rates,N_groups))

# -- symptoms
prop_symptomatic = 0.4531209

# -- recovery
# natural clearance rate (per day)
nat_recovery_rate = 1/76.3540493
# care seeking rate (includes time to recovery after treatment)
care_seeking_rate = 1/15.3720017
# screening rate
screening_rate = 0.3569368/365
# retreatment probability
retreatment_prob_symptomatic = 0.9
# retreatment rate
retreatment_rate_symptomatic = care_seeking_rate/3

# -- strain characteristics
# relative fitness
f_c = 0.98
f_d = 0.98
f_cd = f_c*f_d

# doxypep parameters
doxypep_uptake = 0.5 # vary from 0-0.9
doxypep_rr = 0.38


# --------- set starting conditions
# starting prevalence
scaling_val = 0.03/(.3*0.029+.6*0.154+.1*0.817)
prev_groups_init = [0.817, 0.154, 0.029]

i_init = np.round((np.multiply(N_groups, np.multiply(scaling_val, prev_groups_init))), 0)

# split into symptomatic and asymptomatic
prop_symptomatic_init = 0.1

# split infections by strains
resC_init = 0.0001 # (ceftriaxone resistance)
resD_init = 0.109 # ( doxy resistance)


Y0_init = i_init * prop_symptomatic_init * (1-(resC_init + resD_init - resC_init*resD_init))
Z0_init = i_init * (1-prop_symptomatic_init) * (1-(resC_init + resD_init - resC_init*resD_init))

Yc_init = i_init * prop_symptomatic_init * resC_init * (1-resD_init)
Zc_init = i_init * (1-prop_symptomatic_init) * resC_init * (1-resD_init)

Yd_init = i_init * prop_symptomatic_init * resD_init * (1-resC_init)
Zd_init = i_init * (1-prop_symptomatic_init) * resD_init * (1-resC_init)

Ycd_init = i_init * prop_symptomatic_init * resD_init * resC_init
Zcd_init = i_init * (1-prop_symptomatic_init) * resD_init * resC_init


# exposed (1.6% of infections)
E0_init = i_init * 0.016 * (1-(resC_init + resD_init - resC_init*resD_init))
Ec_init = i_init * 0.016 * resC_init * (1-resD_init)
Ed_init = i_init * 0.016 * resD_init * (1-resC_init)
Ecd_init = i_init * 0.016 * resD_init * resC_init

S_init = N_groups - 1.016*i_init

compartments_inits = np.array([S_init, E0_init, Ec_init, Ed_init, Ecd_init, Y0_init, Z0_init, Yc_init, Zc_init, Yd_init, Zd_init, Ycd_init, Zcd_init])

compartments_inits_flat = compartments_inits.flatten()



# --- params
params_inits = [mu_entry, mu_exit,
                b, K, f_c, f_d, f_cd, 
                nat_recovery_rate, care_seeking_rate, screening_rate, 
                retreatment_prob_symptomatic, retreatment_rate_symptomatic,
                prop_symptomatic,
                doxypep_uptake, doxypep_rr]



# -----------------------------------------------------------------------------
# Run model (for varying doxy-PEP uptake rates)
# -----------------------------------------------------------------------------


for dpval in [10, 30, 50, 70, 90 ]:
    
    doxypep_uptake = dpval/100 # vary from 0.1-0.9
    
        
    # --- params
    params_inits = [mu_entry, mu_exit,
                    b, K, f_c, f_d, f_cd, 
                    nat_recovery_rate, care_seeking_rate, screening_rate, 
                    retreatment_prob_symptomatic, retreatment_rate_symptomatic,
                    prop_symptomatic,
                    doxypep_uptake, doxypep_rr]
    
    nyears = 20
    ts = np.linspace(0, 365*nyears-1, 365*nyears)
    
    result_flat = odeint(ode_wrapper, compartments_inits_flat, ts, args=(params_inits,))
    
    # reshape to (time, compartments, risk groups)
    result = result_flat.reshape((365*nyears,13,3))

    
    # to get the incidences, re-run with original ode fct:
    result_inc = np.zeros((result_flat.shape[0], 8, 3))
    result_care = np.zeros((result_flat.shape[0], 8, 3))
    for i in np.arange(result_flat.shape[0]):
        fool, result_inc[i,:,:], result_care[i,:,:] = ode_model(result_flat[i,:], ts[i], params_inits)
    
    
    
    
    # save simulation dynamics data as dataframe for sampling    
    result_df = pd.DataFrame(result_flat, 
                             columns = ['S h','S m', 'S l','E0 h','E0 m','E0 l', 
                                        'Ec h', 'Ec m', 'Ec l', 'Ed h','Ed m','Ed l', 
                                        'Ecd h','Ecd m','Ecd l', 'Y0 h','Y0 m','Y0 l', 
                                        'Z0 h','Z0 m','Z0 l', 'Yc h','Yc m','Yc l', 
                                        'Zc h','Zc m','Zc l', 'Yd h','Yd m','Yd l', 
                                        'Zd h','Zd m','Zd l', 'Ycd h','Ycd m','Ycd l', 
                                        'Zcd h', 'Zcd m', 'Zcd l'])
    result_df['time'] = ts
    result_df_long = result_df.melt(id_vars = 'time', 
                                     var_name='compartment_group',
                                     value_name='size')
    result_df_long['compartment'] = result_df_long.compartment_group.str.split(' ',expand=True, n=1)[0]
    result_df_long['group'] = result_df_long.compartment_group.str.split(' ',expand=True, n=1)[1]
    # save
    result_df_long.to_csv(outfolder+'sim_doxypep-uptake-'+str(dpval)+'.csv', index=False)
    
    
    # save incidences
    # reshape
    result_inc_df = pd.DataFrame(result_inc.reshape((result_flat.shape[0], 8*3)),
                                 columns = ['new_Y0 h', 'new_Y0 m', 'new_Y0 l', 
                                            'new_Z0 h','new_Z0 m','new_Z0 l', 
                                            'new_Yc h','new_Yc m','new_Yc l', 
                                            'new_Zc h','new_Zc m','new_Zc l', 
                                            'new_Yd h','new_Yd m','new_Yd l', 
                                            'new_Zd h','new_Zd m','new_Zd l', 
                                            'new_Ycd h', 'new_Ycd m', 'new_Ycd l',
                                            'new_Zcd h',  'new_Zcd m', 'new_Zcd l'])
    result_inc_df['time'] = ts
    result_inc_df_long = result_inc_df.melt(id_vars = 'time', 
                                     var_name='new_compartment_group',
                                     value_name='size')
    result_inc_df_long['compartment'] = result_inc_df_long.new_compartment_group.str.split(' ',expand=True, n=1)[0]
    result_inc_df_long['group'] = result_inc_df_long.new_compartment_group.str.split(' ',expand=True, n=1)[1]
    # save
    result_inc_df_long.to_csv(outfolder+'incidences_doxypep-uptake-'+str(dpval)+'.csv', index=False)
    
    
    # save care seekers
    result_care_df = pd.DataFrame(result_care.reshape((result_flat.shape[0], 8*3)),
                                  columns = ['treat_Y0 h', 'treat_Y0 m', 'treat_Y0 l',
                                             'treat_Yc h', 'treat_Yc m', 'treat_Yc l',
                                             'treat_Yd h', 'treat_Yd m', 'treat_Yd l',
                                             'treat_Ycd h', 'treat_Ycd m', 'treat_Ycd l',
                                             'screen_Z0 h', 'screen_Z0 m', 'screen_Z0 l',
                                             'screen_Zc h',  'screen_Zc m',  'screen_Zc l', 
                                             'screen_Zd h',  'screen_Zd m',  'screen_Zd l', 
                                             'screen_Zcd h',  'screen_Zcd m',  'screen_Zcd l', 
                                             ])
    
    result_care_df['time'] = ts
    result_care_df_long = result_care_df.melt(id_vars = 'time', 
                                     var_name='new_compartment_group',
                                     value_name='size')
    result_care_df_long['compartment'] = result_care_df_long.new_compartment_group.str.split(' ',expand=True, n=1)[0]
    result_care_df_long['group'] = result_care_df_long.new_compartment_group.str.split(' ',expand=True, n=1)[1]
    # save
    result_care_df_long.to_csv(outfolder+'care-and-screening_doxypep-uptake-'+str(dpval)+'.csv', index=False)
    
    
        
        
 
        
 
    

