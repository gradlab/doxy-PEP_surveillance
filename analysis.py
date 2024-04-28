#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:37:27 2024

@author: kir312
"""




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns




outfolder = 'output/'
plotfolder = 'plots/'





def calculate_threshold_crossed(dd, percentile, res_threshold, time_var):
    # calculate percentile
    data = dd.groupby([time_var]).doxy_resistance_proportion.quantile((percentile)).reset_index()
    # return min value where proportion is above percentile
    return data[data.doxy_resistance_proportion > res_threshold][time_var].min()


def calculate_delay(dd, true):
    delays = [x-y for x,y in zip(dd, true)]
    return delays






# prep output
# 5th percentile
percentiles_df = pd.DataFrame()

# main outcome: time until threshold is crossed
out_threshold_crossing_time_columns = ['value', 'method', 'pooling_level', 'time_unit', 
                                       'resistance_threshold','dpval', 'sampling_intensity']
out_threshold_crossing_time = pd.DataFrame(columns=out_threshold_crossing_time_columns)
true_threshold_crossing_time = pd.DataFrame()
true_proportions = pd.DataFrame()

gisp_proportion_all = pd.DataFrame()
naats_number_all = pd.DataFrame()

# to compare doxypep uptake levels, prep output

for dpval in  [10, 30, 50, 70, 90 ]:
    
    # -----------------------------------------------------------------------------
    # read incidence data
    # -----------------------------------------------------------------------------
     
    # compartment sizes
    data = pd.read_csv(outfolder+'sim_doxypep-uptake-'+str(dpval)+'.csv')
    
    # incidences
    data_inc = pd.read_csv(outfolder+'incidences_doxypep-uptake-'+str(dpval)+'.csv')
    
    data_inc_wide = data_inc.pivot(columns='new_compartment_group',
                                   index='time',
                                   values='size')
    # monthly incidence data
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
    # read sampling data
    # -----------------------------------------------------------------------------
    
    
    # read sample datasets
    gisp_samples = pd.read_csv(outfolder+'gisp_sample_dpval-'+str(dpval)+'.csv', index_col='Unnamed: 0')
    gisp_samples_multiple = pd.read_csv(outfolder+'gisp_samples_multiple_dpval-'+str(dpval)+'.csv')
    naat_samples = pd.read_csv(outfolder+'naats_sample_dpval-'+str(dpval)+'.csv', index_col='Unnamed: 0')
    naat_samples_multiple = pd.read_csv(outfolder+'naat_samples_multiple-'+str(dpval)+'.csv')

    # add times  (end of month)
    naat_samples_multiple['month'] = [int(x/30) for x in naat_samples_multiple.time]
    naat_samples_multiple['year'] = [int(x/12) for x in naat_samples_multiple.month]
    

    # -----------------------------------------------------------------------------
    # calculate outcomes
    # -----------------------------------------------------------------------------
    
    # *** OUTCOME 1: estimates and 90% confidence intervals for each method
    # culture
    gisp_samples_agg = gisp_samples.groupby(['iteration','month', 
                                             'year']).agg('sum').drop(['time', 'samples_remaining_this_month'], axis=1).reset_index()
    gisp_samples_multiple_agg = gisp_samples_multiple.groupby(['iteration','month', 
                                                               'year', 'n_samples_gisp']).agg('sum').drop(['time', 'samples_remaining_this_month'], axis=1).reset_index()
    
    # naats
    naats_monthly = naat_samples.groupby(['iteration','month', 
                                             'year']).agg('sum').drop('time', axis=1).reset_index()
    
    
    # multiple sampling intensities
    naats_monthly_multiple = naat_samples_multiple.groupby(['iteration','month', 
                                                               'year', 'sampling_proportion']).agg('sum').drop('time', axis=1).reset_index()
    
    # add time in days
    naats_monthly['time'] = (naats_monthly.month+1)*30 
    gisp_samples_agg['time'] = (gisp_samples_agg.month+1)*30 
    
    # calculate total samples 
    for dd in [gisp_samples_agg, gisp_samples_multiple_agg, naats_monthly, naats_monthly_multiple]:
        # sum total samples
        dd['total_samples'] = np.sum(dd.loc[:,['new_samples_0', 'new_samples_c','new_samples_d', 'new_samples_cd']], axis=1)
        
        # calculate proportion
        dd['prop_0'] = dd.new_samples_0 / dd.total_samples
        dd['prop_c'] = dd.new_samples_c / dd.total_samples
        dd['prop_d'] = dd.new_samples_d / dd.total_samples
        dd['prop_cd'] = dd.new_samples_cd / dd.total_samples
        
        dd['doxy_resistance_proportion'] = dd.prop_d + dd.prop_cd
    
    # add variable with resistance proportion (true burden)
    data_inc_wide['doxy_resistance_proportion'] = data_inc_wide.new_d_prop + data_inc_wide.new_cd_prop
    data_inc_monthly['doxy_resistance_proportion'] = data_inc_monthly.new_d_prop + data_inc_monthly.new_cd_prop
    
    
    # calculate percentiles
    
    # -- gisp monthly
    temp_percentile = gisp_samples_agg.groupby(['month']).doxy_resistance_proportion.quantile(([0.05, 0.95])).reset_index()
    temp_mean = gisp_samples_agg.groupby(['month']).agg({'doxy_resistance_proportion':'mean'}).reset_index()
    temp_mean['level_1'] = 'mean'
    temp_percentile = pd.concat((temp_percentile, temp_mean)) 
    temp_percentile['dpval'] = dpval
    temp_percentile['method'] = 'gisp'
    temp_percentile['pooling_level'] = 'monthly'
    temp_percentile['time_unit'] = 'month'
    temp_percentile['sampling_intensity'] = 25
    
    percentiles_df = pd.concat((percentiles_df, temp_percentile))
    
    # -- naats monthly
    temp_percentile = naats_monthly.groupby(['month']).doxy_resistance_proportion.quantile(([0.05, 0.95])).reset_index()
    temp_mean = naats_monthly.groupby(['month']).agg({'doxy_resistance_proportion':'mean'}).reset_index()
    temp_mean['level_1'] = 'mean'
    temp_percentile = pd.concat((temp_percentile, temp_mean)) 
    temp_percentile['dpval'] = dpval
    temp_percentile['method'] = 'naats'
    temp_percentile['pooling_level'] = 'monthly'
    temp_percentile['time_unit'] = 'month'
    temp_percentile['sampling_intensity'] = 0.2
    
    percentiles_df = pd.concat((percentiles_df, temp_percentile))
    
    # true value
    temp_true = data_inc_wide.loc[:,['time', 'doxy_resistance_proportion']]
    temp_true['dpval'] = dpval
    
    true_proportions = pd.concat((true_proportions, temp_true))
                               
    
    # *** OUTCOME 2: time to being 95% confidence that threshold is crossed
    for res_threshold in np.arange(0.11, 0.30, 0.01):
        # true value
        time_threshold_crossed = data_inc_wide[data_inc_wide.doxy_resistance_proportion>res_threshold].time.min()
        
        true_threshold_crossing_time = pd.concat((true_threshold_crossing_time,
                                                  pd.DataFrame({'true_value':time_threshold_crossed, 
                                                                'dpval':dpval, 
                                                                'resistance_threshold':res_threshold}, index=[0])
                                                 ))
        # culture
        m1 = calculate_threshold_crossed(dd=gisp_samples_agg, percentile=0.05, 
                                         res_threshold=res_threshold, time_var='month')
        temp1 = [m1, 'gisp', 'monthly', 'month', res_threshold, dpval, 25]
        # naats  
        #m2 = calculate_threshold_crossed(dd=naats_monthly, percentile=0.05, 
        #                                 res_threshold=res_threshold, time_var='month')
        #temp2 = [m2, 'naats', 'monthly', 'month', res_threshold, dpval, 0.2]
        
        # combine into dataframe
        # columns: method, pooling level, value, resistance threshold, sampling intensity
        tempdf = pd.DataFrame([temp1, #temp2
                               ], columns = out_threshold_crossing_time_columns)
    
        out_threshold_crossing_time = pd.concat((out_threshold_crossing_time, tempdf))
        
    
        # multiple sampling intensities
        templist1 = []
        templist2 = []
        for ss in naats_monthly_multiple.sampling_proportion.unique():
            # naats monthly multiple
            mm1 = calculate_threshold_crossed(dd=naats_monthly_multiple[naats_monthly_multiple.sampling_proportion==ss], 
                                             percentile=0.05, 
                                             res_threshold=res_threshold, time_var='month')
            tempp1 = [mm1, 'naats', 'monthly', 'month', res_threshold, dpval, ss]
            templist1.append(tempp1)
           
        # culture multiple
        for nn in gisp_samples_multiple_agg.n_samples_gisp.unique():
            mm2 = calculate_threshold_crossed(dd=gisp_samples_multiple_agg[gisp_samples_multiple_agg.n_samples_gisp==nn], 
                                             percentile=0.05, 
                                             res_threshold=res_threshold, time_var='month')
            tempp2 = [mm2, 'gisp', 'monthly', 'month', res_threshold, dpval, nn]
            templist2.append(tempp2)
            # naats weekly multiple
        
        
        # add to dataframe
        out_threshold_crossing_time = pd.concat((out_threshold_crossing_time, 
                                                 pd.DataFrame(templist1, columns = out_threshold_crossing_time_columns),
                                                 pd.DataFrame(templist2, columns = out_threshold_crossing_time_columns),
                                                 ))
     
        print(res_threshold)
        
    # -- additional info
    # -- which proportion sampled under gisp?
    gisp_proportion = data_care_wide.loc[:,['time', 'total_observed_infections', 'total_symptomatic_observed_infections']]
    # agg to monthly
    gisp_proportion['month'] = [int(x/30) for x in gisp_proportion.time]
    gisp_proportion.drop(['time'], axis=1, inplace=True)
    gisp_proportion = gisp_proportion.groupby(['month']).agg('sum').reset_index()
    gisp_proportion  = pd.merge(gisp_proportion, gisp_samples_agg.loc[:,['iteration', 'month','total_samples']], 
                                on=['month'], how='left')    
    gisp_proportion['gisp_proportion_symptomatic'] = gisp_proportion.total_samples / gisp_proportion.total_symptomatic_observed_infections
    gisp_proportion['dpval'] = dpval
    gisp_proportion_all = pd.concat((gisp_proportion_all, gisp_proportion))
    
    # -- how many samples for naat proportion?
    naats_number = naats_monthly.loc[:,['iteration', 'month','total_samples']]
    naats_number['dpval'] = dpval
    naats_number['sampling_proportion'] = 0.2
    naats_number_all = pd.concat((naats_number_all, naats_number))
    # at different sampling intensities:
    naats_number_multiple = naats_monthly_multiple.loc[:,['iteration', 'month','total_samples','sampling_proportion']]    
    naats_number_multiple['dpval'] = dpval
    naats_number_all = pd.concat((naats_number_all, naats_number_multiple))
    
    #print(dpval)

    # -------------------------------------------------------------------------------------
    # lineplot with markers for time until 95% confident
    # -------------------------------------------------------------------------------------
    res_thresholds = out_threshold_crossing_time.resistance_threshold.unique()
    true_threshold_crossing_time.resistance_threshold = [round(x, 2) for x in true_threshold_crossing_time.resistance_threshold]

    # lineplot with CI + lollipop overlay for one resistance threshold
    val_true = true_threshold_crossing_time[(true_threshold_crossing_time.dpval==dpval)&
                                            (true_threshold_crossing_time.resistance_threshold==0.15)].true_value[0]
    val_naats = out_threshold_crossing_time[(out_threshold_crossing_time.dpval==dpval)&
                                            (out_threshold_crossing_time.resistance_threshold==res_thresholds[4])&
                                            (out_threshold_crossing_time.method=='naats')&
                                            (out_threshold_crossing_time.sampling_intensity==0.2)]['value'].values[0]
    val_gisp = out_threshold_crossing_time[(out_threshold_crossing_time.dpval==dpval)&
                                            (out_threshold_crossing_time.resistance_threshold==res_thresholds[4])&
                                            (out_threshold_crossing_time.method=='gisp')&
                                            (out_threshold_crossing_time.sampling_intensity==25)]['value'].values[0]
    
    # -- plot
    fig, axs = plt.subplots(1,1, figsize=(7,5))
    sns.lineplot(x=gisp_samples_agg.time, y = gisp_samples_agg.doxy_resistance_proportion,
                 color='purple', ax = axs, errorbar=('pi', 90), zorder=0)
    sns.lineplot(x=naats_monthly.time, y = naats_monthly.doxy_resistance_proportion,
                 color='teal', ax = axs, errorbar=('pi', 90), zorder=0)
    # true incidence
    sns.lineplot(x=data_inc_wide.time, y= data_inc_wide.doxy_resistance_proportion, 
                 color='black', ax = axs)
    
    # horizontal line for threshold
    plt.hlines([0.15], xmin=0, xmax=2*365, color='gray', zorder=0, linewidth=3)
    plt.text(2*365, 0.16,'resistance threshold 0.15',rotation=0, horizontalalignment='right', color='gray')
    # scatters
    sns.scatterplot(x = [val_true], y = [0.15], color='black', label='True level of resistance',
                    marker='o', ax=axs, s=50, zorder=1)
    sns.scatterplot(x = [(val_naats+1)*30], y = [0.15], color='teal', label='NAAT remnants',
                    marker='o', ax=axs, s=50, zorder=1)
    sns.scatterplot(x = [(val_gisp+1)*30], y = [0.15], color='purple', label='Culture',
                    marker='o', ax=axs, s=50, zorder=1)
    axs.set_xlim((0,2*365))
    plt.xlabel('Days')
    plt.ylabel('Resistance proportion')
    axs.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(plotfolder+'lineplot-with-lollipop_sampling_proportions_1st-year_dpval-'+str(dpval)+'_simint90.pdf', dpi=300)
    

        


# save
percentiles_df.to_csv(outfolder+'percentiles_df_monthly.csv', index=False)
gisp_proportion_all.to_csv(outfolder+'gisp_proportion_all.csv', index=False)
naats_number_all.to_csv(outfolder+'naats_number_all.csv', index=False)                                                

    
# *** OUTCOME 3: delay in 95% confidence relative to true time that threshold is crossed

out_threshold_crossing_time.resistance_threshold = [round(x, 2) for x in out_threshold_crossing_time.resistance_threshold]
true_threshold_crossing_time.resistance_threshold = [round(x, 2) for x in true_threshold_crossing_time.resistance_threshold]

# add column with time in days
out_threshold_crossing_time['time_in_days'] = out_threshold_crossing_time.value.copy()
out_threshold_crossing_time['time_in_days'] = np.where(out_threshold_crossing_time.time_unit=='month', 
                                                       (out_threshold_crossing_time.value+1)*30,
                                                       out_threshold_crossing_time.time_in_days)  

# merge true value
out_threshold_crossing_time = pd.merge(out_threshold_crossing_time, 
                                       true_threshold_crossing_time,
                                       on = ['dpval', 'resistance_threshold'],
                                       how='left')

# calculate delay
out_threshold_crossing_time['delay_in_days'] = out_threshold_crossing_time.time_in_days - out_threshold_crossing_time.true_value
    
# add longer method name for plotting
out_threshold_crossing_time['method_name'] = out_threshold_crossing_time.method.replace({'gisp':'Culture',
                                                                                         'naats':'NAAT remnants'})

# save
out_threshold_crossing_time.to_csv(outfolder+'out_threshold_crossing_time.csv', index=False)
true_proportions.to_csv(outfolder+'true_proportions.csv', index=False)    
    
# -----------------------------------------------------------------------------
#  VISUALIZE
# -----------------------------------------------------------------------------
    

colors_multiples = ['#A1E6E6', # lightest teal
                    '#74D1D1', # lighter teal
                    '#00A3A3', # light teal
                    '#008080', # teal
                    '#006666', # dark teal
                    '#DD8ADD', # lightest purple
                    '#B65FB6', # lighter purple
                    '#963E96', # light purple
                    '#742174', # purple
                    '#641164', # medium purple
                    '#4C024C', # dark purple
                    ]

colors_dpvals = ['#D96D9E', # lighter raspberry
                 '#D14D88', # light raspberry
                 '#ae2d68', # raspberry
                 '#82214D', # dark raspberry
                 '#6C0233', # darkest raspberry
                 ]


# prep dataframes with different subsets of methods
data_gisp_naats_monthly = out_threshold_crossing_time[out_threshold_crossing_time.pooling_level=='monthly']


# compare intensities, separate subplots for culture and naats
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
sns.boxplot(y='method_name', x='delay_in_days',
            hue = 'sampling_intensity',
            data = data_gisp_naats_monthly[data_gisp_naats_monthly.method_name=='Culture'],
            ax=axs[0],
            palette = colors_multiples[5:],
            whis=[25, 75],
            showfliers=False,
            )
sns.boxplot(y='method_name', x='delay_in_days',
            hue = 'sampling_intensity',
            data = data_gisp_naats_monthly[data_gisp_naats_monthly.method_name=='NAAT remnants'],
            ax=axs[1],
            palette = colors_multiples[:5],
            whis=[25, 75],
            showfliers=False,
            )
axs[0].spines[['right', 'top']].set_visible(False)
axs[1].spines[['right', 'top']].set_visible(False)
axs[0].set_xlabel('')
axs[1].set_xlabel('Delay (in days)')
axs[0].set_ylabel('Culture')
axs[1].set_ylabel('NAAT remnants')
axs[0].tick_params(axis='y',  which='both', left=False, labelleft=False) 
axs[1].tick_params(axis='y',  which='both', left=False, labelleft=False) 
axs[0].legend(title='Sampling intensity', 
              #bbox_to_anchor=(1.1,1),
              loc='lower right')
axs[1].legend(title='Sampling intensity', #bbox_to_anchor=(1.1,1),
              loc='lower right')
plt.tight_layout()
plt.savefig(plotfolder+'boxplot_2panels_delay_by-method-and-intensity_alldpvals.pdf', dpi=300)
plt.close()    


"""
# in numbers:
# distribution over dpvals and thresholds
# median
data_gisp_naats_monthly.groupby(['method_name', 'sampling_intensity']).agg({'delay_in_days':'median'})
# 75% CI
data_gisp_naats_monthly.groupby(['method_name', 'sampling_intensity']).delay_in_days.quantile(([0.25, 0.75]))

# by dpval
data_gisp_naats_monthly.groupby(['method_name', 'dpval','sampling_intensity']).agg({'delay_in_days':'median'})
"""


 
# 2 panels
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
sns.boxplot(y='method_name', x='delay_in_days',
            hue = 'dpval',
            data = data_gisp_naats_monthly[data_gisp_naats_monthly.method_name=='Culture'],
            ax=axs[0],
            palette = colors_dpvals,
            #palette = colors_multiples[5:],
            whis=[25, 75],
            showfliers=False,
            )
sns.boxplot(y='method_name', x='delay_in_days',
            hue = 'dpval',
            data = data_gisp_naats_monthly[data_gisp_naats_monthly.method_name=='NAAT remnants'],
            ax=axs[1],
            palette = colors_dpvals,
            #palette = colors_multiples[:5],
            whis=[25, 75],
            showfliers=False,
            )
axs[0].spines[['right', 'top']].set_visible(False)
axs[1].spines[['right', 'top']].set_visible(False)
axs[0].set_xlabel('')
axs[1].set_xlabel('Delay (in days)')
axs[0].set_ylabel('Culture')
axs[1].set_ylabel('NAAT remnants')
axs[0].tick_params(axis='y',  which='both', left=False, labelleft=False) 
axs[1].tick_params(axis='y',  which='both', left=False, labelleft=False) 
axs[0].legend(title='Doxy-PEP uptake', 
              #bbox_to_anchor=(1.1,1),
              loc='lower right').remove()
axs[1].legend(title='Doxy-PEP uptake', #bbox_to_anchor=(1.1,1),
              loc='lower right')
plt.tight_layout()
plt.savefig(plotfolder+'boxplot_2panels_delay_by-method-and-dpval_allintensities.pdf', dpi=300)
plt.close()    






# -- average delay in estimate (mean resistance proportion) vs true resistance proportion
delay_estimate = pd.DataFrame()

# calculate time between true proportions and 
for dpval in  [10, 30, 50, 70, 90 ]:

    for tt in np.arange(true_proportions.time.max()):
        temp_true = true_proportions[(true_proportions.dpval==dpval)&(true_proportions.time==tt)]
        temp_true_val = temp_true.doxy_resistance_proportion.values[0]
       
        if temp_true_val<1:
            
            #  culture
            temp_gisp = percentiles_df[(percentiles_df.method=='gisp')&(percentiles_df.level_1=='mean')&(percentiles_df.dpval==dpval)]
            temp_gisp_month = temp_gisp[temp_gisp.doxy_resistance_proportion >= temp_true_val].month.min()
            
            # naats
            temp_naats = percentiles_df[(percentiles_df.method=='naats')&(percentiles_df.level_1=='mean')&(percentiles_df.dpval==dpval)]
            temp_naats_month = temp_naats[temp_naats.doxy_resistance_proportion >= temp_true_val].month.min()
            
            tempout = pd.DataFrame({'res_value':temp_true_val,
                                    'true_time':tt,
                                    'gisp_month':temp_gisp_month,
                                    'gisp_time':(temp_gisp_month+1)*30,
                                    'naats_month':temp_naats_month,
                                    'naats_time':(temp_naats_month+1)*30,
                                    'dpval':dpval
                                    
                }, index=[0])
            
            delay_estimate = pd.concat((delay_estimate, tempout))


delay_estimate['gisp_delay_days'] = delay_estimate.gisp_time - delay_estimate.true_time
delay_estimate['naats_delay_days'] = delay_estimate.naats_time - delay_estimate.true_time
delay_estimate['true_month'] = [int(x/30) for x in delay_estimate.true_time]


# overall

# by dpval, first year
sub = delay_estimate[delay_estimate.true_time < 30*12]

sub.groupby(['dpval']).agg({'gisp_delay_days':'mean',
                                       'naats_delay_days':'mean'})

# first year overall
sub.agg({'gisp_delay_days':'mean','naats_delay_days':'mean'})


# excluding values near 100%
sub2 = delay_estimate[delay_estimate.res_value < 0.8]

sub2.groupby(['dpval']).agg({'gisp_delay_days':'mean',
                                       'naats_delay_days':'mean'})
sub2.agg({'gisp_delay_days':'mean','naats_delay_days':'mean'})


# all time
delay_estimate.groupby(['dpval']).agg({'gisp_delay_days':'mean',
                                       'naats_delay_days':'mean'})


# only end-of-month values
sub3 = delay_estimate[(delay_estimate.true_time%29==0)&(delay_estimate.res_value<0.8)]

sub3.groupby(['dpval']).agg({'gisp_delay_days':'mean',
                                       'naats_delay_days':'mean'})

sub3.agg({'gisp_delay_days':'mean','naats_delay_days':'mean'})


# calculate delay the other way:
# given the estimate of gisp/naats at end of month, when was the true resistance proportion at that level?

# -- average delay in estimate (mean resistance proportion) vs true resistance proportion
delay_estimate_v2 = pd.DataFrame()

# calculate time between true proportions and 
for dpval in  [10, 30, 50, 70, 90 ]:

    for mm in np.arange(243):
        
        #  gisp
        temp_gisp = percentiles_df[(percentiles_df.method=='gisp')&(percentiles_df.level_1=='mean')&(percentiles_df.dpval==dpval)&(percentiles_df.month==mm)]
        temp_gisp_val = temp_gisp.doxy_resistance_proportion.values[0]
        
        # naats
        temp_naats = percentiles_df[(percentiles_df.method=='naats')&(percentiles_df.level_1=='mean')&(percentiles_df.dpval==dpval)&(percentiles_df.month==mm)]
        temp_naats_val = temp_naats.doxy_resistance_proportion.values[0]
        
        # true time when estimated resistance was attained
        temp_true = true_proportions[(true_proportions.dpval==dpval)]
        
        if temp_gisp_val<1.0:
            temp_true_day_gisp = temp_true[temp_true.doxy_resistance_proportion >= temp_gisp_val].time.min()
            delay_gisp_in_days = (mm+1)*30 - temp_true_day_gisp
            true_month_gisp = int(temp_true_day_gisp/30)
        else:
            temp_true_day_gisp = np.nan
            delay_gisp_in_days = np.nan
            true_month_gisp = np.nan
        
        if temp_naats_val<1.0:
            temp_true_day_naats = temp_true[temp_true.doxy_resistance_proportion >= temp_naats_val].time.min()
            delay_naats_in_days = (mm+1)*30 - temp_true_day_naats
            true_month_naats = int(temp_true_day_naats/30)
        else:
            temp_true_day_naats = np.nan
            delay_naats_in_days = np.nan
            true_month_naats = np.nan
        
        tempout = pd.DataFrame({'month':mm,
                                'gisp_val':temp_gisp_val,
                                'naats_val':temp_naats_val,
                                'dpval':dpval,
                                'true_day_gisp':temp_true_day_gisp,
                                'true_day_naats':temp_true_day_naats,
                                'true_month_gisp': true_month_gisp,
                                'true_month_naats': true_month_naats,
                                'delay_gisp_in_days':delay_gisp_in_days,
                                'delay_naats_in_days': delay_naats_in_days,
                                }, index=[0])
        
        delay_estimate_v2 = pd.concat((delay_estimate_v2, tempout))

# overall 
delay_estimate_v2.groupby(['dpval']).agg({'delay_gisp_in_days':'mean',
                                       'delay_naats_in_days':'mean'})

# excluding values near 100%
sub2 = delay_estimate_v2[delay_estimate_v2.gisp_val < 0.8]
sub2.groupby('dpval').agg({'delay_gisp_in_days':'mean'})
sub2.agg({'delay_gisp_in_days':'mean'})

sub3 = delay_estimate_v2[delay_estimate_v2.naats_val < 0.8]
sub3.groupby('dpval').agg({'delay_naats_in_days':'mean'})
sub3.agg({'delay_naats_in_days':'mean'})



# average difference between gisp and naats
delay_estimate_v2['gisp_naat_difference'] = delay_estimate_v2.gisp_val - delay_estimate_v2.naats_val
# absolute difference
np.mean(np.abs(delay_estimate_v2['gisp_naat_difference']))

# for example in paper (dpval 30):
# resistance proportion at 6 months
true_proportions[(true_proportions.dpval==30)&(true_proportions.time==6*30)] 
# 19.3%
# gisp estimate:
percentiles_df[(percentiles_df.method=='gisp')&(percentiles_df.dpval==30)&(percentiles_df.month==6)].loc[:,['doxy_resistance_proportion', 'level_1']]
# 0.18836
# naats:
percentiles_df[(percentiles_df.method=='naats')&(percentiles_df.dpval==30)&(percentiles_df.month==6)].loc[:,['doxy_resistance_proportion', 'level_1']]
# 0.191338






