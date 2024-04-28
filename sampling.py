#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:44:19 2024

@author: kir312
"""  


import numpy as np
import pandas as pd




outfolder = 'output/'



# -----------------------------------------------------------------------------
# sampling #1 - culture
# -----------------------------------------------------------------------------


# option 1: first n_samples_gisp selected randomly from all symptomatic urethral samples
def generate_gisp_sample(usedata, prop_symptomatic_urethral, n_samples_gisp):

    m = 0 # month indicator
    t = 0
    gisp_sample = pd.DataFrame()
    while t < usedata.shape[0]:
        # reset the 25 samples
        avail_samples = n_samples_gisp
        # loop over days in each month
        for d in np.arange(30):
            # reset the counts
            samples_0 = 0
            samples_c = 0
            samples_d = 0
            samples_cd = 0
            
            
            # define the number that are urethral
            urethral_0 = np.random.binomial(usedata.treat_Y0[t], prop_symptomatic_urethral)
            urethral_c = np.random.binomial(usedata.treat_Yc[t], prop_symptomatic_urethral)
            urethral_d = np.random.binomial(usedata.treat_Yd[t], prop_symptomatic_urethral)
            urethral_cd = np.random.binomial(usedata.treat_Ycd[t], prop_symptomatic_urethral)
            
            total_new_symptomatic_urethral = urethral_0 + urethral_c + urethral_d + urethral_cd
            
            # if no more avail samples for the month:
            if avail_samples == 0:
                samples_0 = 0
                samples_c = 0
                samples_d = 0
                samples_cd = 0
        
            # if total new infections per day are < available samples: select all
            # add some stochasticity to the samples that are urethral
            elif total_new_symptomatic_urethral < avail_samples:
                samples_0 = urethral_0
                samples_c = urethral_c
                samples_d = urethral_d
                samples_cd = urethral_cd
                
            else:  
            # else, calculate proportion of infections from each strain
            # stochastic sampling to determine the number (int) of samples selected from each strain
                samples_0, samples_c, samples_d, samples_cd, not_urethral = np.random.multinomial(n=avail_samples, 
                                                                                    pvals=[usedata.treat_Y0_prop_sympt[t] * prop_symptomatic_urethral, 
                                                                                           usedata.treat_Yc_prop_sympt[t] * prop_symptomatic_urethral,
                                                                                           usedata.treat_Yd_prop_sympt[t] * prop_symptomatic_urethral, 
                                                                                           usedata.treat_Ycd_prop_sympt[t] * prop_symptomatic_urethral,
                                                                                           (1-prop_symptomatic_urethral)])
    
                
            temp = pd.DataFrame({'time':t,
                                 'month':m,
                                 'year':int(m/12),
                                 'new_samples_0':samples_0,
                                 'new_samples_c':samples_c,
                                 'new_samples_d':samples_d,
                                 'new_samples_cd':samples_cd,
                                 'samples_remaining_this_month':avail_samples
                                 }, index=[t])
            gisp_sample = pd.concat((gisp_sample,temp))
            
            # remove "used" samples from stock
            avail_samples = avail_samples - (samples_0 + samples_c + samples_d + samples_cd)
            # add a day to the day counter
            t += 1
            if t==usedata.shape[0]: break
        # update month counter
        m += 1 
        
    return gisp_sample


def generate_multiple_gisp_samples(usedata, prop_symptomatic_urethral, n_samples_gisp, n_iter):
    
    all_gisp_samples = pd.DataFrame()
    for i in np.arange(n_iter):
        gisp_sample = generate_gisp_sample(usedata, prop_symptomatic_urethral, n_samples_gisp)
        gisp_sample['iteration'] = i
        all_gisp_samples = pd.concat((all_gisp_samples, gisp_sample))
    
    return all_gisp_samples
        


# -----------------------------------------------------------------------------
# sampling #2 - NAAT remnants
# -----------------------------------------------------------------------------

# random sample of all positive tests (symptomatic and asymptomatic)

def generate_naats_sample(usedata, sampling_prop):
    
    naat_sample = pd.DataFrame()
    
    for t in np.arange(usedata.shape[0]):
       
        samples_0, samples_c, samples_d, samples_cd = np.random.multinomial(n = (usedata.total_observed_infections[t] * sampling_prop).astype(int), 
                                                                            pvals=[usedata.treat_Y0_prop[t] + usedata.screen_Z0_prop[t],
                                                                                   usedata.treat_Yc_prop[t] + usedata.screen_Zc_prop[t],
                                                                                   usedata.treat_Yd_prop[t] + usedata.screen_Zd_prop[t],
                                                                                   usedata.treat_Ycd_prop[t] + usedata.screen_Zcd_prop[t]])

        temp = pd.DataFrame({'time':t,
                             'new_samples_0':samples_0,
                             'new_samples_c':samples_c,
                             'new_samples_d':samples_d,
                             'new_samples_cd':samples_cd,
                             }, index=[t])
        naat_sample = pd.concat((naat_sample, temp))
    
    
    return naat_sample
    


def generate_multiple_naat_samples(usedata, sampling_prop, n_iter):
    
    all_naat_samples = pd.DataFrame()
    for i in np.arange(n_iter):
        naat_sample = generate_naats_sample(usedata, sampling_prop)
        naat_sample['iteration'] = i
        all_naat_samples = pd.concat((all_naat_samples, naat_sample))
    
    return all_naat_samples









# -----------------------------------------------------------------------------
# Run sampling
# -----------------------------------------------------------------------------


# prep output

alldpvals_times_true = []
alldpvals_months_naats = []
alldpvals_months_gisp = []
alldpvals_res_thresholds = []
alldpvals_month_naats_multiple = pd.DataFrame()
alldpvals_month_gisp_multiple = pd.DataFrame()

# delays
alldpvals_months_naats_delay = []
alldpvals_months_gisp_delay = []



for dpval in  [10, 30, 50, 70, 90 ]:
    
    
    # -----------------------------------------------------------------------------
    # read simulation data
    # -----------------------------------------------------------------------------

    # compartment sizes
    data = pd.read_csv(outfolder+'sim_doxypep-uptake-'+str(dpval)+'.csv')
    
    # incidences
    data_inc = pd.read_csv(outfolder+'incidences_doxypep-uptake-'+str(dpval)+'.csv')
    
    data_inc_wide = data_inc.pivot(columns='new_compartment_group',
                                   index='time',
                                   values='size')
    # make monthly incidence data
    data_inc_monthly = data_inc.copy()
    data_inc_monthly = data_inc_monthly.groupby(['time', 'compartment']).agg({'size':'sum'}).reset_index()
    data_inc_monthly = data_inc_monthly.pivot(columns='compartment',
                                   index='time',
                                   values='size')
    data_inc_monthly.reset_index(inplace=True)
    data_inc_monthly['month'] = [int(x/30) for x in data_inc_monthly.time]
    data_inc_monthly = data_inc_monthly.groupby(['month']).agg('sum').drop('time', axis=1).reset_index()
    
    
    # care seeking (treatment, retreatment, and screening)
    data_care = pd.read_csv(outfolder+'care-and-screening_doxypep-uptake-'+str(dpval)+'.csv')
    data_care_agg = data_care.groupby(['time', 'compartment']).agg({'size':'sum'}).reset_index()
    data_care_wide = data_care_agg.pivot(columns='compartment',
                                   index='time',
                                   values='size')
    
    
    # -----------------------------------------------------------------------------
    # feature engineering
    # -----------------------------------------------------------------------------
    
    # --- incidences
    # total infections
    data_inc_wide['total_incident_infections'] = np.sum(data_inc_wide, axis=1)
    
    # infections by strain - symptomatic
    data_inc_wide['new_Y0'] = np.sum(data_inc_wide.loc[:,['new_Y0 h', 'new_Y0 m', 'new_Y0 l']], axis=1)
    data_inc_wide['new_Yc'] = np.sum(data_inc_wide.loc[:,['new_Yc h','new_Yc m','new_Yc l']], axis=1)
    data_inc_wide['new_Yd'] = np.sum(data_inc_wide.loc[:,['new_Yd h','new_Yd m','new_Yd l']], axis=1)
    data_inc_wide['new_Ycd'] = np.sum(data_inc_wide.loc[:,['new_Ycd h', 'new_Ycd m', 'new_Ycd l']], axis=1)
    
    # infections by strain - asymptomatic
    data_inc_wide['new_Z0'] = np.sum(data_inc_wide.loc[:,['new_Z0 h', 'new_Z0 m', 'new_Z0 l']], axis=1)
    data_inc_wide['new_Zc'] = np.sum(data_inc_wide.loc[:,['new_Zc h','new_Zc m','new_Zc l']], axis=1)
    data_inc_wide['new_Zd'] = np.sum(data_inc_wide.loc[:,['new_Zd h','new_Zd m','new_Zd l']], axis=1)
    data_inc_wide['new_Zcd'] = np.sum(data_inc_wide.loc[:,['new_Zcd h', 'new_Zcd m', 'new_Zcd l']], axis=1)
    
    # strain totals
    data_inc_wide['new_0'] = data_inc_wide.new_Y0 + data_inc_wide.new_Z0
    data_inc_wide['new_c'] = data_inc_wide.new_Yc + data_inc_wide.new_Zc
    data_inc_wide['new_d'] = data_inc_wide.new_Yd + data_inc_wide.new_Zd
    data_inc_wide['new_cd'] = data_inc_wide.new_Ycd + data_inc_wide.new_Zcd
    
    # proportions
    data_inc_wide['new_Y0_prop'] = data_inc_wide.new_Y0 / data_inc_wide.total_incident_infections
    data_inc_wide['new_Yc_prop'] = data_inc_wide.new_Yc / data_inc_wide.total_incident_infections
    data_inc_wide['new_Yd_prop'] = data_inc_wide.new_Yd / data_inc_wide.total_incident_infections
    data_inc_wide['new_Ycd_prop'] = data_inc_wide.new_Ycd / data_inc_wide.total_incident_infections
    data_inc_wide['new_Z0_prop'] = data_inc_wide.new_Z0 / data_inc_wide.total_incident_infections
    data_inc_wide['new_Zc_prop'] = data_inc_wide.new_Zc / data_inc_wide.total_incident_infections
    data_inc_wide['new_Zd_prop'] = data_inc_wide.new_Zd / data_inc_wide.total_incident_infections
    data_inc_wide['new_Zcd_prop'] = data_inc_wide.new_Zcd / data_inc_wide.total_incident_infections
    data_inc_wide['new_0_prop'] = data_inc_wide.new_0 / data_inc_wide.total_incident_infections
    data_inc_wide['new_c_prop'] = data_inc_wide.new_c / data_inc_wide.total_incident_infections
    data_inc_wide['new_d_prop'] = data_inc_wide.new_d / data_inc_wide.total_incident_infections
    data_inc_wide['new_cd_prop'] = data_inc_wide.new_cd / data_inc_wide.total_incident_infections
    
    # total symptomatic
    data_inc_wide['new_symptomatic'] = np.sum(data_inc_wide.loc[:,['new_Y0', 'new_Yc', 'new_Yd', 'new_Ycd']], axis=1)
    data_inc_wide['new_asymptomatic'] = np.sum(data_inc_wide.loc[:,['new_Z0', 'new_Zc', 'new_Zd', 'new_Zcd']], axis=1)
    
    # proportions of all symptomatic
    data_inc_wide['new_Y0_prop_symptomatic'] = data_inc_wide.new_Y0 / data_inc_wide.new_symptomatic
    data_inc_wide['new_Yc_prop_symptomatic'] = data_inc_wide.new_Yc / data_inc_wide.new_symptomatic
    data_inc_wide['new_Yd_prop_symptomatic'] = data_inc_wide.new_Yd / data_inc_wide.new_symptomatic
    data_inc_wide['new_Ycd_prop_symptomatic'] = data_inc_wide.new_Ycd / data_inc_wide.new_symptomatic
    
    
    # -- monthly data
    # total infections
    data_inc_monthly['total_incident_infections'] = np.sum(data_inc_monthly.drop(['month'], axis=1), axis=1)
    
    # strain totals
    data_inc_monthly['new_0'] = data_inc_monthly.new_Y0 + data_inc_monthly.new_Z0
    data_inc_monthly['new_c'] = data_inc_monthly.new_Yc + data_inc_monthly.new_Zc
    data_inc_monthly['new_d'] = data_inc_monthly.new_Yd + data_inc_monthly.new_Zd
    data_inc_monthly['new_cd'] = data_inc_monthly.new_Ycd + data_inc_monthly.new_Zcd
    
    # relevant proportions
    data_inc_monthly['new_0_prop'] = data_inc_monthly.new_0 / data_inc_monthly.total_incident_infections
    data_inc_monthly['new_c_prop'] = data_inc_monthly.new_c / data_inc_monthly.total_incident_infections
    data_inc_monthly['new_d_prop'] = data_inc_monthly.new_d / data_inc_monthly.total_incident_infections
    data_inc_monthly['new_cd_prop'] = data_inc_monthly.new_cd / data_inc_monthly.total_incident_infections
    
    
    
    
    # --- care seekers
    # total observed infections
    data_care_wide['total_observed_infections'] = np.sum(data_care_wide, axis=1)
    # total symptomatic observed infections
    data_care_wide['total_symptomatic_observed_infections'] = np.sum(data_care_wide.loc[:,['treat_Yc', 'treat_Ycd','treat_Y0',
                                                                                           'treat_Yd']], axis=1)
    
    
    # proportions - relative to total observed infections
    data_care_wide['treat_Y0_prop'] = data_care_wide.treat_Y0 / data_care_wide.total_observed_infections
    data_care_wide['treat_Yc_prop'] = data_care_wide.treat_Yc / data_care_wide.total_observed_infections
    data_care_wide['treat_Yd_prop'] = data_care_wide.treat_Yd / data_care_wide.total_observed_infections
    data_care_wide['treat_Ycd_prop'] = data_care_wide.treat_Ycd / data_care_wide.total_observed_infections
    data_care_wide['screen_Z0_prop'] = data_care_wide.screen_Z0 / data_care_wide.total_observed_infections
    data_care_wide['screen_Zc_prop'] = data_care_wide.screen_Zc / data_care_wide.total_observed_infections
    data_care_wide['screen_Zd_prop'] = data_care_wide.screen_Zd / data_care_wide.total_observed_infections
    data_care_wide['screen_Zcd_prop'] = data_care_wide.screen_Zcd / data_care_wide.total_observed_infections
    
    # proportions - relative to total symptomatic
    data_care_wide['treat_Y0_prop_sympt'] = data_care_wide.treat_Y0 / data_care_wide.total_symptomatic_observed_infections
    data_care_wide['treat_Yc_prop_sympt'] = data_care_wide.treat_Yc / data_care_wide.total_symptomatic_observed_infections
    data_care_wide['treat_Yd_prop_sympt'] = data_care_wide.treat_Yd / data_care_wide.total_symptomatic_observed_infections
    data_care_wide['treat_Ycd_prop_sympt'] = data_care_wide.treat_Ycd / data_care_wide.total_symptomatic_observed_infections
    
    
     
    data_inc_wide.reset_index(inplace=True)
    data_care_wide.reset_index(inplace=True)
    


    
    # -----------------------------------------------------------------------------
    # generate samples
    # -----------------------------------------------------------------------------
    
    
    usedata = data_care_wide.copy()
    
    
    n_iter = 1000
    
    
    # ----- culture samples
    n_samples_gisp = 25 # monthly
    prop_symptomatic_urethral = 0.8
    
    # sample:
    gisp_samples = generate_multiple_gisp_samples(usedata, prop_symptomatic_urethral, n_samples_gisp, n_iter)
    
    # save
    gisp_samples.to_csv(outfolder+'gisp_sample_dpval-'+str(dpval)+'.csv')
    
    
    
    
    # ----- naat sample
    sampling_prop = 0.2
    
    # sample:
    naat_samples = generate_multiple_naat_samples(usedata, sampling_prop, n_iter)
    
    # add month and year variables to naat sample
    naat_samples['month'] = [int(x/30) for x in naat_samples.time]
    naat_samples['year'] = [int(x/12) for x in naat_samples.month]
    
    # save
    naat_samples.to_csv(outfolder+'naats_sample_dpval-'+str(dpval)+'.csv')
    
    

    
    # -----------------------------------------------------------------------------
    # compare different levels of sampling
    # -----------------------------------------------------------------------------
    
    # -- GISP: comparison of multiple numbers of samples
    gisp_samples_multiple = pd.DataFrame()
    
    for n_samples_gisp in [5, 10, 20, 40, 80]:
        temp_sample = generate_multiple_gisp_samples(usedata, prop_symptomatic_urethral, n_samples_gisp, n_iter)
        
        temp_sample['n_samples_gisp'] = n_samples_gisp
        
        gisp_samples_multiple = pd.concat((gisp_samples_multiple, temp_sample))
    
    # save
    gisp_samples_multiple.to_csv(outfolder+'gisp_samples_multiple_dpval-'+str(dpval)+'.csv', index=False)
    
        
    
    # -- NAATS: comparison of multiple sampling proportions
    naat_samples_multiple = pd.DataFrame()
    
    for sampling_prop in [0.05, 0.1, 0.2, 0.4, 0.8]:
        
        temp_sample = generate_multiple_naat_samples(usedata, sampling_prop, n_iter)
        
        temp_sample['sampling_proportion'] = sampling_prop
        
        naat_samples_multiple = pd.concat((naat_samples_multiple, temp_sample))
    
    # save    
    naat_samples_multiple.to_csv(outfolder+'naat_samples_multiple-'+str(dpval)+'.csv', index=False)
    
    


    
    
    


    
    
    
    
    
    

















